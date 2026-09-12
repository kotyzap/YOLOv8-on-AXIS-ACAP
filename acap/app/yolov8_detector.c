/**
 * YOLOv8 Detector for AXIS cameras - main loop, decode and overlay.
 *
 * Derived from the "object-detection-yolov5" example in
 * AxisCommunications/acap-native-sdk-examples:
 *     Copyright (C) 2025, Axis Communications AB, Lund, Sweden (Apache-2.0)
 * Those portions remain under Apache-2.0.
 *
 * The YOLOv8 decode path, the threaded pipeline, the live settings, the event
 * sender and the live-view JSON are new:
 *     Copyright (C) 2026 Pavel Kotyza <kotyza@gmail.com>
 *
 * As part of this application the file is distributed under AGPL-3.0, because
 * the application ships an Ultralytics YOLOv8-derived model. See LICENSE.
 *
 * Not an Axis Communications product.
 */

/**
 * YOLOv8 detector for ARTPEC-8.
 *
 * Derived from Axis's object-detection-yolov5 example. The decode path is
 * rewritten for YOLOv8's channel-major, objectness-free, two-tensor output:
 * value c of detection i lives at [c * num_detections + i], boxes are
 * cx,cy,w,h in model-input pixels, scores are already sigmoid'd per class.
 *
 * The DLPU and the CPU run concurrently: frame N is decoded on a worker
 * thread while frame N+1 is on the DLPU, so the loop is inference-bound.
 * The cost is that overlays describe the previous frame -- one frame of lag.
 *
 * Arguments: MODELFILE LABELSFILE [-c DEVICE]
 */

#include "argparse.h"
#include "channel_util.h"
#include "events.h"
#include "img_util.h"
#include "labelparse.h"
#include "model.h"
#include "model_params.h"  // Generated at build time
#include "panic.h"
#include "vdo-error.h"
#include "vdo-frame.h"
#include "vdo-types.h"
#include <axsdk/axparameter.h>
#include <bbox.h>

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>

#include <poll.h>
#include <unistd.h>

#define APP_NAME "yolov8_detector"

// Hard floor on the confidence threshold. At 0 every anchor survives and the
// O(n^2) NMS would run over all NUM_DETECTIONS of them, wedging the app.
#define MIN_CONF_THRESHOLD 0.01f

// Second guard on the same problem: never compare more than this many boxes,
// however low the threshold goes. Ultralytics' max_det default is 300.
#define MAX_NMS_CANDIDATES 300

// Poll timeout, so a stalled stream still lets SIGTERM through.
#define POLL_TIMEOUT_MS 1000

#define LIVE_PATH            "/usr/local/packages/" APP_NAME "/html/live.json"
#define LIVE_MIN_INTERVAL_MS 1000
// The settings page re-arms every 30 s while it is visible; twice that is the
// window in which the app keeps writing after the last time it heard from it.
#define LIVE_ARM_MS          60000

volatile sig_atomic_t running = 1;

static void shutdown(int status) {
    (void)status;
    running = 0;
}

typedef struct model_params {
    int input_width;
    int input_height;
    float box_scale;  // quantization of the 4 x num_detections box tensor
    float box_zero_point;
    float score_scale;  // quantization of the num_classes x num_detections score tensor
    float score_zero_point;
    int num_classes;
    int num_detections;
} model_params_t;

static const model_params_t model_params = {
    .input_width      = MODEL_INPUT_WIDTH,
    .input_height     = MODEL_INPUT_HEIGHT,
    .box_scale        = BOX_QUANT_SCALE,
    .box_zero_point   = BOX_QUANT_ZERO_POINT,
    .score_scale      = SCORE_QUANT_SCALE,
    .score_zero_point = SCORE_QUANT_ZERO_POINT,
    .num_classes      = NUM_CLASSES,
    .num_detections   = NUM_DETECTIONS,
};

static inline float box_at(const uint8_t* t, int c, int i, const model_params_t* mp) {
    return (t[c * mp->num_detections + i] - mp->box_zero_point) * mp->box_scale;
}

static inline float score_from_q(int q, const model_params_t* mp) {
    return (q - mp->score_zero_point) * mp->score_scale;
}

// Per-anchor working set. Single-threaded ownership: everything below is
// written by the decode worker and read by the main thread only while the
// worker is idle.
static uint8_t decode_best_q[NUM_DETECTIONS];
static int16_t decode_best_class[NUM_DETECTIONS];
static int invalid_detections[NUM_DETECTIONS];
static float detection_confidence[NUM_DETECTIONS];
static int detection_label[NUM_DETECTIONS];

typedef struct {
    int index;
    float confidence;
} ranked_t;

static ranked_t ranked[NUM_DETECTIONS];

// A frame handed to the decode worker: its own copy of the output tensors and
// of every setting, so the main thread can start the next inference and the
// parameter callbacks can keep running without racing the decode.
typedef struct {
    uint8_t* boxes;
    uint8_t* scores;
    float conf_threshold;
    float iou_threshold;
    float max_area;
    int class_allowed[NUM_CLASSES];
} decode_job_t;

static decode_job_t decode_job;
static pthread_t decode_thread;
static pthread_mutex_t decode_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t decode_cv    = PTHREAD_COND_INITIALIZER;
static int decode_pending          = 0;
static int decode_quit             = 0;

static int ax_parameter_get_int(AXParameter* handle, const char* name, int fallback) {
    gchar* str_value = NULL;
    GError* error    = NULL;
    int value        = fallback;

    if (!ax_parameter_get(handle, name, &str_value, &error)) {
        syslog(LOG_WARNING,
               "Parameter %s unreadable (%s); using %d",
               name,
               error ? error->message : "unknown",
               fallback);
        g_clear_error(&error);
        return fallback;
    }

    if (sscanf(str_value, "%d", &value) != 1) {
        syslog(LOG_WARNING, "Parameter %s is not an int ('%s'); using %d", name, str_value, fallback);
        value = fallback;
    } else {
        syslog(LOG_INFO, "Parameter %s: %s", name, str_value);
    }

    g_free(str_value);
    return value;
}

// Caller frees the returned string. Returns a copy of fallback if unreadable.
static gchar* ax_parameter_get_str(AXParameter* handle, const char* name, const char* fallback) {
    gchar* str_value = NULL;
    GError* error    = NULL;

    if (!ax_parameter_get(handle, name, &str_value, &error)) {
        syslog(LOG_WARNING,
               "Parameter %s unreadable (%s); using '%s'",
               name,
               error ? error->message : "unknown",
               fallback);
        g_clear_error(&error);
        return g_strdup(fallback);
    }

    syslog(LOG_INFO, "Parameter %s: '%s'", name, str_value);
    return str_value;
}

static float clamp_conf(int percent) {
    float v = percent / 100.0f;
    return v < MIN_CONF_THRESHOLD ? MIN_CONF_THRESHOLD : v;
}

// Builds allowed[] from a comma-separated list of label names. A blank list means
// every class is allowed. Unknown names are reported and ignored.
static void build_class_mask(const char* list,
                             char** labels,
                             size_t num_labels,
                             int* allowed,
                             int num_classes) {
    int blank = 1;
    for (const char* p = list; *p != '\0'; p++) {
        if (!g_ascii_isspace(*p)) {
            blank = 0;
            break;
        }
    }

    if (blank) {
        for (int c = 0; c < num_classes; c++) {
            allowed[c] = 1;
        }
        syslog(LOG_INFO, "Class filter off: all %d classes active", num_classes);
        return;
    }

    for (int c = 0; c < num_classes; c++) {
        allowed[c] = 0;
    }

    int active    = 0;
    gchar** names = g_strsplit(list, ",", -1);
    for (int i = 0; names[i] != NULL; i++) {
        gchar* name = g_strstrip(names[i]);
        if (*name == '\0') {
            continue;
        }

        int found = 0;
        for (size_t c = 0; c < num_labels && (int)c < num_classes; c++) {
            if (g_ascii_strcasecmp(name, labels[c]) == 0) {
                if (!allowed[c]) {
                    allowed[c] = 1;
                    active++;
                }
                found = 1;
                break;
            }
        }
        if (!found) {
            syslog(LOG_WARNING, "Class filter: no label named '%s', ignored", name);
        }
    }
    g_strfreev(names);

    if (active == 0) {
        syslog(LOG_WARNING,
               "Class filter matched no labels; falling back to all %d classes",
               num_classes);
        for (int c = 0; c < num_classes; c++) {
            allowed[c] = 1;
        }
        return;
    }

    syslog(LOG_INFO, "Class filter on: %d of %d classes active", active, num_classes);
}

// Settings the app re-reads while running. Saving in the web page updates the
// axparameters, the parameter library calls back, and the next frame uses the new
// values -- no restart, no model reload. Only the main thread touches this; the
// decode worker gets a snapshot in decode_job_t.
typedef struct {
    float conf_threshold;
    float iou_threshold;
    float max_area;
    int events_enabled;
    int colour_by_class;  // 0 draws every box red, as before
    gint64 live_until_us;  // stop writing live.json once the page goes quiet
    int class_allowed[NUM_CLASSES];
    char** labels;
    size_t num_labels;
    int num_classes;
    event_sender_t* event_sender;
} settings_t;

static void on_parameter_changed(const gchar* name, const gchar* value, gpointer data) {
    settings_t* st = data;

    int number = atoi(value);  // -Wbad-function-cast: convert via a variable, not a cast

    if (g_str_has_suffix(name, "ConfThresholdPercent")) {
        st->conf_threshold = clamp_conf(number);
    } else if (g_str_has_suffix(name, "IouThresholdPercent")) {
        st->iou_threshold = number / 100.0f;
    } else if (g_str_has_suffix(name, "MaxAreaPercent")) {
        st->max_area = number / 100.0f;
    } else if (g_str_has_suffix(name, "EventsEnabled")) {
        st->events_enabled = (g_ascii_strcasecmp(value, "yes") == 0);
    } else if (g_str_has_suffix(name, "EventMinDurationMs")) {
        event_sender_set_timing(st->event_sender, number, -1);
    } else if (g_str_has_suffix(name, "EventCooldownMs")) {
        event_sender_set_timing(st->event_sender, -1, number);
    } else if (g_str_has_suffix(name, "BoxColours")) {
        st->colour_by_class = (g_ascii_strcasecmp(value, "class") == 0);
    } else if (g_str_has_suffix(name, "LiveView")) {
        // The settings page arms this while it is open and visible, and writes 0
        // when hidden. Without it the app writes nothing to flash.
        //
        // It carries a nonce rather than a yes/no because the page has to re-arm
        // on a timer, and a parameter set to the value it already holds may not
        // produce a change callback at all -- in which case the window would
        // expire under an open page. A value that differs every time cannot.
        st->live_until_us =
            (number != 0) ? g_get_monotonic_time() + (gint64)LIVE_ARM_MS * 1000 : 0;
    } else if (g_str_has_suffix(name, "Classes")) {
        build_class_mask(value, st->labels, st->num_labels, st->class_allowed, st->num_classes);
    } else {
        return;
    }

    syslog(LOG_INFO, "Setting changed: %s = '%s'", name, value);
}

// Writes the current detections to a small JSON file inside the package's html
// directory, where the settings page can poll it. Only while the page says it is
// watching, at most once a second, and never when the payload has not changed --
// this is flash, not a socket.
static void live_write(const char* json, gint64 now, gint64 live_until_us) {
    static gint64 last_us  = 0;
    static char* last_json = NULL;
    static int failed      = 0;

    if (failed || now >= live_until_us) {
        return;
    }

    if (last_us != 0 && (now - last_us) / 1000 < LIVE_MIN_INTERVAL_MS) {
        return;
    }
    if (last_json != NULL && strcmp(last_json, json) == 0) {
        return;  // nothing changed; leave the flash alone
    }

    GError* error = NULL;
    if (!g_file_set_contents(LIVE_PATH, json, -1, &error)) {
        syslog(LOG_WARNING,
               "Live view: cannot write %s (%s); disabling",
               LIVE_PATH,
               error ? error->message : "unknown");
        g_clear_error(&error);
        failed = 1;
        return;
    }

    last_us = now;
    g_free(last_json);
    last_json = g_strdup(json);
}

// Draw on one view. bbox numbers views the way VAPIX does (view area 1 is the first),
// so the ViewArea parameter maps straight through; 0 means "leave it to the app" and
// draws on view 1, the full view on a camera with no view areas configured.
static bbox_t* setup_bbox(unsigned int view) {
    bbox_t* bbox = bbox_view_new((bbox_channel_t)(view ? view : 1u));
    if (!bbox) {
        panic("Failed to create box drawer");
    }

    bbox_clear(bbox);

    bbox_style_outline(bbox);
    bbox_thickness_thin(bbox);

    return bbox;
}

// Box colour by what the class is, so a frame with people and cars reads without
// labels -- bbox draws rectangles only, no text. The same four colours mark the
// groups in the settings page's class list, which is the whole point: the colour
// is the bond between the picture and the list.
//
// COCO order fixes the ranges: 0 person, 1..8 the vehicles, 14..23 the animals.
// Everything else is one bucket; splitting it further would need a table, and
// four colours is already as many as an operator can hold at a glance.
static bbox_color_t class_colour(int class_index, int by_class) {
    if (!by_class) {
        return bbox_color_from_rgb(0xff, 0x00, 0x00);  // the original single red
    }
    if (class_index == 0) {
        return bbox_color_from_rgb(0x22, 0xc5, 0x5e);  // people: green
    }
    if (class_index >= 1 && class_index <= 8) {
        return bbox_color_from_rgb(0x3b, 0x82, 0xf6);  // vehicles: blue
    }
    if (class_index >= 14 && class_index <= 23) {
        return bbox_color_from_rgb(0xa8, 0x55, 0xf7);  // animals: purple
    }
    return bbox_color_from_rgb(0xf9, 0x73, 0x16);      // everything else: amber
}

static float intersection_over_union(float x1,
                                     float y1,
                                     float w1,
                                     float h1,
                                     float x2,
                                     float y2,
                                     float w2,
                                     float h2) {
    float xx1 = fmaxf(x1 - (w1 / 2), x2 - (w2 / 2));
    float yy1 = fmaxf(y1 - (h1 / 2), y2 - (h2 / 2));
    float xx2 = fminf(x1 + (w1 / 2), x2 + (w2 / 2));
    float yy2 = fminf(y1 + (h1 / 2), y2 + (h2 / 2));

    float inter_area = fmaxf(0.0f, xx2 - xx1) * fmaxf(0.0f, yy2 - yy1);
    float union_area = w1 * h1 + w2 * h2 - inter_area;

    if (union_area <= 0.0f) {
        return 0.0f;  // a degenerate box suppresses nothing
    }
    return inter_area / union_area;
}

static int ranked_cmp(const void* a, const void* b) {
    const float ca = ((const ranked_t*)a)->confidence;
    const float cb = ((const ranked_t*)b)->confidence;
    if (ca < cb) return 1;  // descending
    if (ca > cb) return -1;
    return 0;
}

static void
find_corners(float x, float y, float w, float h, float* x1, float* y1, float* x2, float* y2) {
    *x1 = fmaxf(0.0f, x - (w / 2));
    *y1 = fmaxf(0.0f, y - (h / 2));
    *x2 = fminf(1.0f, x + (w / 2));
    *y2 = fminf(1.0f, y + (h / 2));
}

// Box coordinates come out in model-input pixels, so divide by the input size to get
// the frame-normalized 0..1 the bbox API expects.
static void determine_bbox_coordinates(const uint8_t* boxes,
                                       int i,
                                       const model_params_t* mp,
                                       float* x1,
                                       float* y1,
                                       float* x2,
                                       float* y2) {
    float x = box_at(boxes, 0, i, mp) / (float)mp->input_width;
    float y = box_at(boxes, 1, i, mp) / (float)mp->input_height;
    float w = box_at(boxes, 2, i, mp) / (float)mp->input_width;
    float h = box_at(boxes, 3, i, mp) / (float)mp->input_height;
    find_corners(x, y, w, h, x1, y1, x2, y2);
}

// Fraction of the frame a detection's clamped box covers. Negative when the box
// lies wholly outside the frame: cx can reach 657 px on a 640 px input, and
// find_corners then clamps x1 above x2. Callers must treat <= 0 as "reject",
// not as "small".
static float box_area_fraction(const uint8_t* boxes, int i, const model_params_t* mp) {
    float x1, y1, x2, y2;
    determine_bbox_coordinates(boxes, i, mp, &x1, &y1, &x2, &y2);
    return (x2 - x1) * (y2 - y1);
}

// Greedy NMS over the surviving detections, strongest first, per class.
//
// Three things matter here. Working strongest-first is what NMS means: the best
// box wins its neighbourhood, rather than whichever anchor happens to come first
// in the buffer. Gathering the survivors before comparing keeps the O(n^2) over
// the handful that passed the confidence test instead of over all the anchors,
// and MAX_NMS_CANDIDATES bounds even that. Comparing only within a class is what
// Ultralytics does by default: a backpack on a person is a separate detection,
// not a duplicate of the person.
static void non_maximum_suppression(const uint8_t* boxes,
                                    const float* confidence,
                                    const int* label,
                                    float iou_threshold,
                                    const model_params_t* mp,
                                    int* invalid) {
    int n = 0;
    for (int i = 0; i < mp->num_detections; i++) {
        if (!invalid[i]) {
            ranked[n].index      = i;
            ranked[n].confidence = confidence[i];
            n++;
        }
    }

    qsort(ranked, (size_t)n, sizeof(ranked_t), ranked_cmp);

    // Anything past the cap is not worth keeping and is certainly not worth
    // comparing against everything else.
    for (int a = MAX_NMS_CANDIDATES; a < n; a++) {
        invalid[ranked[a].index] = 1;
    }
    if (n > MAX_NMS_CANDIDATES) {
        syslog(LOG_DEBUG, "NMS: %d candidates capped to %d", n, MAX_NMS_CANDIDATES);
        n = MAX_NMS_CANDIDATES;
    }

    for (int a = 0; a < n; a++) {
        const int i = ranked[a].index;
        if (invalid[i]) {
            continue;
        }

        const float x1 = box_at(boxes, 0, i, mp);
        const float y1 = box_at(boxes, 1, i, mp);
        const float w1 = box_at(boxes, 2, i, mp);
        const float h1 = box_at(boxes, 3, i, mp);

        for (int b = a + 1; b < n; b++) {
            const int j = ranked[b].index;
            if (invalid[j] || label[j] != label[i]) {
                continue;
            }

            const float x2 = box_at(boxes, 0, j, mp);
            const float y2 = box_at(boxes, 1, j, mp);
            const float w2 = box_at(boxes, 2, j, mp);
            const float h2 = box_at(boxes, 3, j, mp);

            // i is at least as confident as j, so i always wins.
            if (intersection_over_union(x1, y1, w1, h1, x2, y2, w2, h2) > iou_threshold) {
                invalid[j] = 1;
            }
        }
    }
}

// Smallest quantized score that still clears the threshold, so the hot loop can
// compare raw bytes and never touch a float. Returns 256 when nothing can pass.
static int score_threshold_q(float conf, const model_params_t* mp) {
    double q = ceil((double)conf / (double)mp->score_scale + (double)mp->score_zero_point);
    if (q < 0.0) {
        return 0;
    }
    if (q > 255.0) {
        return 256;
    }
    return (int)q;
}

// Best class per anchor. Iterating class-outer/anchor-inner walks each class row
// sequentially instead of striding num_detections bytes per read, which is what
// the cache wants; the comparison stays in the quantized domain.
static void find_best_class(const uint8_t* scores,
                            const int* class_allowed,
                            const model_params_t* mp) {
    int zp = (int)mp->score_zero_point;
    if (zp < 0) {
        zp = 0;
    }
    if (zp > 255) {
        zp = 255;
    }

    memset(decode_best_q, (unsigned char)zp, sizeof(uint8_t) * (size_t)mp->num_detections);
    memset(decode_best_class, 0, sizeof(int16_t) * (size_t)mp->num_detections);

    for (int c = 0; c < mp->num_classes; c++) {
        if (!class_allowed[c]) {
            continue;
        }
        const uint8_t* row = scores + (size_t)c * (size_t)mp->num_detections;
        for (int i = 0; i < mp->num_detections; i++) {
            if (row[i] > decode_best_q[i]) {
                decode_best_q[i]     = row[i];
                decode_best_class[i] = (int16_t)c;
            }
        }
    }
}

// Fills confidence[] and label[] with the best class per anchor, marks anchors below
// conf_threshold invalid, then runs NMS over what is left.
static void filter_detections(const decode_job_t* job, const model_params_t* mp) {
    const int thr_q = score_threshold_q(job->conf_threshold, mp);

    find_best_class(job->scores, job->class_allowed, mp);

    for (int i = 0; i < mp->num_detections; i++) {
        const int q = decode_best_q[i];

        detection_label[i]      = decode_best_class[i];
        detection_confidence[i] = score_from_q(q, mp);

        if (q < thr_q) {
            invalid_detections[i] = 1;
            continue;
        }
        // Reject frame-sized boxes: YOLOv8n fires a stable full-frame anchor on close,
        // cluttered scenes that NMS cannot suppress (its IoU with the real boxes is low).
        // Reject empty and off-frame ones too -- a negative area is not a small area,
        // and an inverted rectangle is not something to hand to the box drawer.
        const float area = box_area_fraction(job->boxes, i, mp);
        invalid_detections[i] = (area <= 0.0f || area > job->max_area) ? 1 : 0;
    }

    non_maximum_suppression(job->boxes,
                            detection_confidence,
                            detection_label,
                            job->iou_threshold,
                            mp,
                            invalid_detections);
}

static void* decode_worker(void* arg) {
    (void)arg;

    pthread_mutex_lock(&decode_lock);
    for (;;) {
        while (!decode_pending && !decode_quit) {
            pthread_cond_wait(&decode_cv, &decode_lock);
        }
        if (decode_quit) {
            break;
        }
        pthread_mutex_unlock(&decode_lock);

        filter_detections(&decode_job, &model_params);

        pthread_mutex_lock(&decode_lock);
        decode_pending = 0;
        pthread_cond_broadcast(&decode_cv);
    }
    pthread_mutex_unlock(&decode_lock);
    return NULL;
}

static void decode_wait_idle(void) {
    pthread_mutex_lock(&decode_lock);
    while (decode_pending) {
        pthread_cond_wait(&decode_cv, &decode_lock);
    }
    pthread_mutex_unlock(&decode_lock);
}

static void decode_submit(void) {
    pthread_mutex_lock(&decode_lock);
    decode_pending = 1;
    pthread_cond_broadcast(&decode_cv);
    pthread_mutex_unlock(&decode_lock);
}

static int handle_vdo_failed(GError* error) {
    // Maintenance/Installation in progress (e.g. Global-Rotation)
    if (vdo_error_is_expected(&error)) {
        syslog(LOG_INFO, "Expected vdo error %s", error->message);
        return EXIT_SUCCESS;
    } else {
        panic("Unexpected vdo error %s", error->message);
    }
}

static VdoStream* create_new_vdo_stream(unsigned int channel,
                                        VdoFormat format,
                                        VdoResolution res,
                                        unsigned int num_buffers,
                                        const char* image_fit,
                                        double framerate) {
    g_autoptr(VdoMap) vdo_settings = vdo_map_new();
    g_autoptr(GError) error        = NULL;

    if (!vdo_settings) {
        panic("%s: Failed to create vdo_map", __func__);
    }

    vdo_map_set_uint32(vdo_settings, "channel", channel);
    vdo_map_set_uint32(vdo_settings, "format", format);
    vdo_map_set_double(vdo_settings, "framerate", framerate);
    VdoPair32u resolution = {
        .w = res.width,
        .h = res.height,
    };
    vdo_map_set_pair32u(vdo_settings, "resolution", resolution);
    // Make it possible to change the framerate for the stream after it is started
    vdo_map_set_boolean(vdo_settings, "dynamic.framerate", true);

    // The number of buffers that vdo will allocate for this stream. Normally two
    // buffers are enough and using too many buffers will use more memory.
    vdo_map_set_uint32(vdo_settings, "buffer.count", num_buffers);
    // vdo_stream_get_buffer is non blocking and returns immediately, so poll first.
    vdo_map_set_boolean(vdo_settings, "socket.blocking", false);
    vdo_map_set_string(vdo_settings, "image.fit", image_fit);

    g_autoptr(VdoStream) vdo_stream = vdo_stream_new(vdo_settings, NULL, &error);
    if (!vdo_stream) {
        panic("%s: Failed creating vdo stream: %s", __func__, error->message);
    }

    return g_steal_pointer(&vdo_stream);
}

int main(int argc, char** argv) {
    bbox_t* bbox                          = NULL;
    g_autoptr(GError) vdo_error           = NULL;
    model_provider_t* model_provider      = NULL;
    model_tensor_output_t* tensor_outputs = NULL;
    img_info_t model_metadata             = {0};
    img_framerate_t image_framerate       = {0};
    g_autoptr(VdoStream) vdo_stream       = NULL;
    g_autoptr(VdoMap) vdo_stream_info     = NULL;
    const model_params_t* mp              = &model_params;

    signal(SIGTERM, shutdown);
    signal(SIGINT, shutdown);

    args_t args;
    parse_args(argc, argv, &args);

    syslog(LOG_INFO, "YOLOv8 Detector version " APP_VERSION);
    syslog(LOG_INFO, "Model input w/h: %d x %d", mp->input_width, mp->input_height);
    syslog(LOG_INFO,
           "Quantization: box scale %f zp %f, score scale %f zp %f",
           (double)mp->box_scale,
           (double)mp->box_zero_point,
           (double)mp->score_scale,
           (double)mp->score_zero_point);
    syslog(LOG_INFO, "Classes: %d, anchors: %d", mp->num_classes, mp->num_detections);

    GError* axparameter_error       = NULL;
    AXParameter* axparameter_handle = ax_parameter_new(APP_NAME, &axparameter_error);
    if (axparameter_handle == NULL) {
        panic("%s", axparameter_error->message);
    }

    float conf_threshold = clamp_conf(ax_parameter_get_int(axparameter_handle, "ConfThresholdPercent", 25));
    float iou_threshold  = ax_parameter_get_int(axparameter_handle, "IouThresholdPercent", 45) / 100.0f;
    float max_area       = ax_parameter_get_int(axparameter_handle, "MaxAreaPercent", 90) / 100.0f;
    gchar* class_list    = ax_parameter_get_str(axparameter_handle, "Classes", "person");

    gchar* events_enabled_str = ax_parameter_get_str(axparameter_handle, "EventsEnabled", "yes");
    int events_enabled        = (g_ascii_strcasecmp(events_enabled_str, "yes") == 0);
    int event_min_duration_ms = ax_parameter_get_int(axparameter_handle, "EventMinDurationMs", 1000);
    int event_cooldown_ms     = ax_parameter_get_int(axparameter_handle, "EventCooldownMs", 30000);
    int view_area             = ax_parameter_get_int(axparameter_handle, "ViewArea", 0);
    gchar* box_colours_str    = ax_parameter_get_str(axparameter_handle, "BoxColours", "class");
    int colour_by_class       = (g_ascii_strcasecmp(box_colours_str, "class") == 0);
    g_free(box_colours_str);
    g_free(events_enabled_str);

    event_sender_t* event_sender = NULL;

    size_t number_output_tensors = 0;
    model_provider = model_provider_new(args.model_file, args.device_name, &number_output_tensors);
    if (!model_provider) {
        panic("%s: Could not create model provider", __func__);
    }

    tensor_outputs = calloc(number_output_tensors, sizeof(model_tensor_output_t));
    if (!tensor_outputs) {
        panic("%s: Could not allocate tensor outputs", __func__);
    }

    model_metadata = model_provider_get_model_metadata(model_provider);

    double vdo_stream_framerate          = 30.0;
    channel_util_log_channels();
    // ViewArea 0 keeps the historical behaviour: detect on the first input channel,
    // i.e. the sensor's full view. Any other value selects that view area as both the
    // source of frames and the surface the boxes are drawn on, so the two cannot drift
    // apart.
    //
    // A ViewArea that VDO does not know -- a view area deleted after it was chosen,
    // or a number from a camera with different numbering -- must not be fatal: with
    // runMode respawn, a panic here is a restart loop that the settings page cannot
    // reach, because the app it talks to is never up. Fall back to the full view and
    // say so, loudly enough to be found.
    unsigned int vdo_channel = channel_util_get_first_input_channel();
    if (view_area > 0) {
        if (channel_util_channel_exists((unsigned int)view_area)) {
            vdo_channel = (unsigned int)view_area;
        } else {
            syslog(LOG_WARNING,
                   "Parameter ViewArea: %d is not a channel VDO knows about; "
                   "detecting on the full view (channel %u) instead. "
                   "Fix it in the settings page or clear it with param.cgi.",
                   view_area,
                   vdo_channel);
            view_area = 0;
        }
    }
    syslog(LOG_INFO,
           "Parameter ViewArea: %d (detecting and drawing on VDO channel %u)",
           view_area,
           vdo_channel);
    unsigned int vdo_stream_buffer_count = 2;

    uint32_t rotation = channel_util_get_image_rotation(vdo_channel);
    syslog(LOG_INFO, "[Channel %u] Current global rotation is %u", vdo_channel, rotation);
    VdoPair32u channel_ar = channel_util_get_aspect_ratio(vdo_channel);
    syslog(LOG_INFO,
           "[Channel %u] Current aspect ratio is %u:%u",
           vdo_channel,
           channel_ar.w,
           channel_ar.h);

    VdoResolution req_res    = {model_metadata.width, model_metadata.height};
    VdoResolution chosen_req = req_res;

    if (!channel_util_choose_stream_resolution(vdo_channel,
                                               req_res,
                                               &chosen_req,
                                               rotation,
                                               &model_metadata.format)) {
        panic("%s: Could not chose a resolution", __func__);
    }
    syslog(LOG_INFO,
           "Requested %ux%u from vdo for a %dx%d model input",
           chosen_req.width,
           chosen_req.height,
           mp->input_width,
           mp->input_height);

    vdo_stream = create_new_vdo_stream(vdo_channel,
                                       model_metadata.format,
                                       chosen_req,
                                       vdo_stream_buffer_count,
                                       "crop",
                                       vdo_stream_framerate);
    if (!vdo_stream) {
        return handle_vdo_failed(vdo_error);
    }
    vdo_stream_info = vdo_stream_get_info(vdo_stream, &vdo_error);
    if (!vdo_stream_info) {
        return handle_vdo_failed(vdo_error);
    }

    image_framerate.wanted_framerate = vdo_stream_framerate;
    double info_framerate = vdo_map_get_double(vdo_stream_info, "framerate", vdo_stream_framerate);
    image_framerate.frametime = (unsigned int)((1.0 / info_framerate) * 1000.0);

    int fd = vdo_stream_get_fd(vdo_stream, &vdo_error);
    if (fd < 0) {
        return handle_vdo_failed(vdo_error);
    }
    struct pollfd fds = {
        .fd     = fd,
        .events = POLL_IN,
    };

    char** labels         = NULL;
    size_t num_labels     = 0;
    char* label_file_data = NULL;

    parse_labels(&labels, &label_file_data, args.labels_file, &num_labels);

    // Always create the sender, even when events are switched off, so the switch can
    // be flipped later without a restart. Declaring costs nothing while idle.
    event_sender = event_sender_new(mp->num_classes, event_min_duration_ms, event_cooldown_ms);
    if (!events_enabled) {
        syslog(LOG_INFO, "Events disabled by axparameter");
    }

    settings_t settings = {
        .conf_threshold = conf_threshold,
        .iou_threshold  = iou_threshold,
        .max_area       = max_area,
        .events_enabled = events_enabled,
        .colour_by_class = colour_by_class,
        .live_until_us  = 0,
        .labels         = labels,
        .num_labels     = num_labels,
        .num_classes    = mp->num_classes,
        .event_sender   = event_sender,
    };

    build_class_mask(class_list, labels, num_labels, settings.class_allowed, mp->num_classes);
    g_free(class_list);

    static const char* live_params[] = {"ConfThresholdPercent",
                                        "IouThresholdPercent",
                                        "MaxAreaPercent",
                                        "Classes",
                                        "EventsEnabled",
                                        "EventMinDurationMs",
                                        "EventCooldownMs",
                                        "BoxColours",
                                        "LiveView"};
    for (size_t i = 0; i < G_N_ELEMENTS(live_params); i++) {
        GError* cb_error = NULL;
        if (!ax_parameter_register_callback(axparameter_handle,
                                            live_params[i],
                                            on_parameter_changed,
                                            &settings,
                                            &cb_error)) {
            syslog(LOG_WARNING,
                   "Cannot watch %s (%s); it will need a restart to take effect",
                   live_params[i],
                   cb_error ? cb_error->message : "unknown");
            g_clear_error(&cb_error);
        }
    }
    syslog(LOG_INFO, "Settings are applied live; saving does not restart the app");

    bbox = setup_bbox(vdo_channel);

    if (!vdo_stream_start(vdo_stream, &vdo_error)) {
        return handle_vdo_failed(vdo_error);
    }
    syslog(LOG_INFO, "Start fetching video frames from VDO");

    model_provider_update_image_metadata(model_provider, vdo_stream_info);

    // larod does not guarantee the order of the output tensors, so identify them by
    // size: boxes are num_classes/4 times smaller than scores.
    if (number_output_tensors != 2) {
        panic("Expected 2 output tensors (boxes, scores), model has %zu", number_output_tensors);
    }

    const size_t boxes_bytes  = (size_t)(4 * mp->num_detections);
    const size_t scores_bytes = (size_t)mp->num_classes * (size_t)mp->num_detections;
    decode_job.boxes          = malloc(boxes_bytes);
    decode_job.scores         = malloc(scores_bytes);
    if (!decode_job.boxes || !decode_job.scores) {
        panic("%s: Could not allocate decode buffers", __func__);
    }
    for (int i = 0; i < mp->num_detections; i++) {
        invalid_detections[i] = 1;
    }

    if (pthread_create(&decode_thread, NULL, decode_worker, NULL) != 0) {
        panic("%s: Could not start the decode thread", __func__);
    }

    float best_per_class[NUM_CLASSES];
    int have_results       = 0;
    gint64 last_frame_us   = 0;
    unsigned int period_ms = 0;

    while (running) {
        int status = 0;
        do {
            status = poll(&fds, 1, POLL_TIMEOUT_MS);
        } while (status == -1 && errno == EINTR && running);

        if (!running) {
            break;
        }
        if (status < 0) {
            panic("Failed to poll with status %d", status);
        }
        if (status == 0) {
            continue;  // no frame within the timeout; loop so SIGTERM is seen
        }

        g_autoptr(VdoBuffer) vdo_buf = vdo_stream_get_buffer(vdo_stream, &vdo_error);
        if (!vdo_buf && g_error_matches(vdo_error, VDO_ERROR, VDO_ERROR_NO_DATA)) {
            g_clear_error(&vdo_error);
            continue;
        }
        if (!vdo_buf) {
            return handle_vdo_failed(vdo_error);
        }

        const gint64 now = g_get_monotonic_time();
        period_ms        = last_frame_us ? (unsigned int)((now - last_frame_us) / 1000) : 0;
        last_frame_us    = now;

        // Runs on the DLPU. The worker is decoding the previous frame meanwhile.
        if (!model_run_inference(model_provider, vdo_buf)) {
            if (!img_util_flush(vdo_stream, &vdo_buf, &vdo_error)) {
                return handle_vdo_failed(vdo_error);
            }
            continue;
        }

        decode_wait_idle();

        // Publish the previous frame's results before its buffers are overwritten.
        if (have_results) {
            bbox_clear(bbox);

            for (int c = 0; c < mp->num_classes; c++) {
                best_per_class[c] = 0.0f;
            }

            GString* live = g_string_new(NULL);
            // "view" is the view area the app actually runs on -- 0 for the full view --
            // which differs from the saved parameter after a fallback. The page streams
            // whatever this says, so the picture and the list come from one source.
            g_string_append_printf(live,
                                   "{\"fps\":%.1f, \"view\":%d, \"detections\":[",
                                   period_ms ? 1000.0 / (double)period_ms : 0.0,
                                   view_area);
            int live_count = 0;
            int shown      = 0;

            for (int i = 0; i < mp->num_detections; i++) {
                if (invalid_detections[i]) {
                    continue;
                }
                shown++;

                if (detection_confidence[i] > best_per_class[detection_label[i]]) {
                    best_per_class[detection_label[i]] = detection_confidence[i];
                }

                float x1, y1, x2, y2;
                determine_bbox_coordinates(decode_job.boxes, i, mp, &x1, &y1, &x2, &y2);

                syslog(LOG_DEBUG,
                       "Object %d: Label=%s, Confidence=%.2f, Box=[%.2f, %.2f, %.2f, %.2f]",
                       shown,
                       labels[detection_label[i]],
                       (double)detection_confidence[i],
                       (double)x1,
                       (double)y1,
                       (double)x2,
                       (double)y2);

                if (live_count < 16) {
                    g_string_append_printf(
                        live,
                        "%s{\"label\":\"%s\",\"confidence\":%.2f,\"box\":[%.3f,%.3f,%.3f,%.3f]}",
                        live_count ? "," : "",
                        labels[detection_label[i]],
                        (double)detection_confidence[i],
                        (double)x1,
                        (double)y1,
                        (double)x2,
                        (double)y2);
                    live_count++;
                }

                // Frame-normalized space is aligned with the frame VDO delivers, rotation
                // included: verified on the Q1656 at rotation 180 by comparing live.json
                // against a snapshot with the overlay. No compensation is needed here.
                // A digital zoom on the camera, however, crops the *displayed* stream while
                // the detector sees the full channel, so boxes drift under zoom.
                bbox_coordinates_frame_normalized(bbox);
                bbox_color(bbox, class_colour(detection_label[i], settings.colour_by_class));
                bbox_rectangle(bbox, x1, y1, x2, y2);
            }

            if (!bbox_commit(bbox, 0u)) {
                panic("Failed to commit box drawer");
            }

            if (settings.events_enabled) {
                event_sender_update(event_sender, best_per_class, labels, num_labels);
            }

            g_string_append(live, "]}");
            live_write(live->str, now, settings.live_until_us);
            g_string_free(live, TRUE);
        }

        // Drives the axparameter change callbacks and the event system's own
        // bookkeeping. Nothing else iterates the default context in this app.
        while (g_main_context_iteration(NULL, FALSE)) {
        }

        // Snapshot this frame's tensors and settings, then let the worker decode
        // them while the next inference runs.
        for (size_t i = 0; i < number_output_tensors; i++) {
            if (!model_get_tensor_output_info(model_provider, i, &tensor_outputs[i])) {
                panic("Failed to get output tensor info for %zu", i);
            }
        }
        const uint8_t* boxes  = NULL;
        const uint8_t* scores = NULL;
        for (size_t i = 0; i < number_output_tensors; i++) {
            if (tensor_outputs[i].size == boxes_bytes) {
                boxes = tensor_outputs[i].data;
            } else if (tensor_outputs[i].size == scores_bytes) {
                scores = tensor_outputs[i].data;
            }
        }
        if (!boxes || !scores) {
            panic("Could not identify box and score tensors by size");
        }

        memcpy(decode_job.boxes, boxes, boxes_bytes);
        memcpy(decode_job.scores, scores, scores_bytes);
        decode_job.conf_threshold = settings.conf_threshold;
        decode_job.iou_threshold  = settings.iou_threshold;
        decode_job.max_area       = settings.max_area;
        memcpy(decode_job.class_allowed,
               settings.class_allowed,
               sizeof(int) * (size_t)mp->num_classes);

        decode_submit();
        have_results = 1;

        // Feed the true frame period to the framerate governor: with the decode
        // overlapped, that is what the pipeline actually sustains.
        if (period_ms && img_util_update_framerate(vdo_stream, &image_framerate, period_ms)) {
            if (!img_util_flush(vdo_stream, &vdo_buf, &vdo_error)) {
                return handle_vdo_failed(vdo_error);
            }
        } else {
            if (!vdo_stream_buffer_unref(vdo_stream, &vdo_buf, &vdo_error)) {
                if (!vdo_error_is_expected(&vdo_error)) {
                    panic("%s: Unexpected error: %s", __func__, vdo_error->message);
                }
                g_clear_error(&vdo_error);
            }
        }
    }

    pthread_mutex_lock(&decode_lock);
    decode_quit = 1;
    pthread_cond_broadcast(&decode_cv);
    pthread_mutex_unlock(&decode_lock);
    pthread_join(decode_thread, NULL);

    free(decode_job.boxes);
    free(decode_job.scores);
    event_sender_free(event_sender);
    ax_parameter_free(axparameter_handle);
    if (model_provider) {
        model_provider_destroy(model_provider);
    }
    free(tensor_outputs);
    free(labels);
    free(label_file_data);
    bbox_destroy(bbox);

    syslog(LOG_INFO, "Exit %s", argv[0]);

    return 0;
}
