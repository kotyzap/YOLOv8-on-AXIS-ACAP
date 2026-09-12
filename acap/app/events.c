/**
 * YOLOv8 Detector for AXIS cameras
 * Copyright (C) 2026 Pavel Kotyza <kotyza@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version. See the LICENSE file that ships with this application.
 *
 * Not an Axis Communications product.
 */

#include "events.h"

#include <axsdk/axevent.h>
#include <stdlib.h>
#include <syslog.h>

#define TOPIC_NS   "tnsaxis"
#define TOPIC0     "CameraApplicationPlatform"
#define TOPIC1     "YOLOv8Detector"
#define TOPIC2     "Detection"
#define KEY_CLASS  "class"
#define KEY_CONF   "confidence"

// A class that blinks out for a single frame has not left the scene. Without
// this, one missed detection restarts the min-duration clock and a marginal
// object never reaches the threshold.
#define ABSENCE_GRACE_MS 500

struct event_sender {
    AXEventHandler* handler;
    guint declaration;
    int num_classes;
    int min_duration_ms;
    int cooldown_ms;
    gint64* above_since_us;  // 0 when the class is not currently present
    gint64* last_sent_us;    // 0 when nothing has been sent yet
    gint64* last_seen_us;    // when the class was last present at all
    int send_failed_logged;  // so a broken event system logs once, not per frame
    int registered;  // the event system confirms the declaration asynchronously
};

static void declaration_complete(guint declaration, gpointer user_data) {
    (void)declaration;
    event_sender_t* es = user_data;
    es->registered     = 1;
    syslog(LOG_INFO, "Events: declaration registered");
}

// Topics are identical for the declaration and for every event sent, so both
// paths build them here.
static gboolean add_topics(AXEventKeyValueSet* set) {
    return ax_event_key_value_set_add_key_value(set, "topic0", TOPIC_NS, TOPIC0, AX_VALUE_TYPE_STRING, NULL) &&
           ax_event_key_value_set_add_key_value(set, "topic1", TOPIC_NS, TOPIC1, AX_VALUE_TYPE_STRING, NULL) &&
           ax_event_key_value_set_add_key_value(set, "topic2", TOPIC_NS, TOPIC2, AX_VALUE_TYPE_STRING, NULL);
}

event_sender_t* event_sender_new(int num_classes, int min_duration_ms, int cooldown_ms) {
    GError* error                = NULL;
    AXEventKeyValueSet* set      = NULL;
    event_sender_t* es           = NULL;
    const gdouble zero_conf      = 0.0;

    set = ax_event_key_value_set_new();
    if (!add_topics(set) ||
        !ax_event_key_value_set_add_key_value(set, KEY_CLASS, NULL, "", AX_VALUE_TYPE_STRING, &error) ||
        !ax_event_key_value_set_add_key_value(set, KEY_CONF, NULL, &zero_conf, AX_VALUE_TYPE_DOUBLE, &error) ||
        !ax_event_key_value_set_mark_as_data(set, KEY_CLASS, NULL, &error) ||
        !ax_event_key_value_set_mark_as_data(set, KEY_CONF, NULL, &error)) {
        syslog(LOG_WARNING,
               "Events: could not describe the event (%s); continuing without events",
               error ? error->message : "unknown");
        goto fail;
    }

    es = calloc(1, sizeof(*es));
    if (es == NULL) {
        goto fail;
    }

    es->handler = ax_event_handler_new();
    if (!ax_event_handler_declare(es->handler, set, TRUE, &es->declaration, declaration_complete, es, &error)) {
        syslog(LOG_WARNING,
               "Events: declaration refused (%s); continuing without events",
               error ? error->message : "unknown");
        ax_event_handler_free(es->handler);
        free(es);
        es = NULL;
        goto fail;
    }

    es->num_classes     = num_classes;
    es->min_duration_ms = min_duration_ms;
    es->cooldown_ms     = cooldown_ms;
    es->above_since_us  = calloc((size_t)num_classes, sizeof(gint64));
    es->last_sent_us    = calloc((size_t)num_classes, sizeof(gint64));
    es->last_seen_us    = calloc((size_t)num_classes, sizeof(gint64));
    if (es->above_since_us == NULL || es->last_sent_us == NULL ||
        es->last_seen_us == NULL) {
        event_sender_free(es);
        es = NULL;
        goto fail;
    }

    syslog(LOG_INFO,
           "Events: declared %s:%s/%s:%s/%s:%s, min duration %d ms, cooldown %d ms",
           TOPIC_NS, TOPIC0, TOPIC_NS, TOPIC1, TOPIC_NS, TOPIC2,
           min_duration_ms, cooldown_ms);

fail:
    ax_event_key_value_set_free(set);
    g_clear_error(&error);
    return es;
}

void event_sender_free(event_sender_t* es) {
    if (es == NULL) {
        return;
    }
    if (es->handler != NULL) {
        ax_event_handler_undeclare(es->handler, es->declaration, NULL);
        ax_event_handler_free(es->handler);
    }
    free(es->above_since_us);
    free(es->last_sent_us);
    free(es->last_seen_us);
    free(es);
}

static void send_one(event_sender_t* es, const char* label, double confidence) {
    GError* error           = NULL;
    AXEventKeyValueSet* set = ax_event_key_value_set_new();

    if (ax_event_key_value_set_add_key_value(set, KEY_CLASS, NULL, label, AX_VALUE_TYPE_STRING, &error) &&
        ax_event_key_value_set_add_key_value(set, KEY_CONF, NULL, &confidence, AX_VALUE_TYPE_DOUBLE, &error)) {
        AXEvent* event = ax_event_new2(set, NULL);

        if (ax_event_handler_send_event(es->handler, es->declaration, event, &error)) {
            syslog(LOG_INFO, "Event sent: class=%s, confidence=%.2f", label, confidence);
        } else if (!es->send_failed_logged) {
            syslog(LOG_WARNING,
                   "Events: send failed (%s); no further send failures will be logged",
                   error ? error->message : "unknown");
            es->send_failed_logged = 1;
        }

        ax_event_free(event);
    }

    ax_event_key_value_set_free(set);
    g_clear_error(&error);
}

void event_sender_set_timing(event_sender_t* es, int min_duration_ms, int cooldown_ms) {
    if (es == NULL) {
        return;
    }
    if (min_duration_ms >= 0) {
        es->min_duration_ms = min_duration_ms;
    }
    if (cooldown_ms >= 0) {
        es->cooldown_ms = cooldown_ms;
    }
}

void event_sender_update(event_sender_t* es,
                         const float* best_per_class,
                         char** labels,
                         size_t num_labels) {
    if (es == NULL) {
        return;
    }

    if (!es->registered) {
        return;
    }

    const gint64 now = g_get_monotonic_time();

    for (int c = 0; c < es->num_classes; c++) {
        if (best_per_class[c] <= 0.0f) {
            // Only really gone once it has been missing for the whole grace window.
            if (es->above_since_us[c] != 0 && es->last_seen_us[c] != 0 &&
                (now - es->last_seen_us[c]) / 1000 > ABSENCE_GRACE_MS) {
                es->above_since_us[c] = 0;  // the next appearance starts over
            }
            continue;
        }

        es->last_seen_us[c] = now;
        if (es->above_since_us[c] == 0) {
            es->above_since_us[c] = now;
        }

        const gint64 held_ms  = (now - es->above_since_us[c]) / 1000;
        const gint64 since_ms = (now - es->last_sent_us[c]) / 1000;

        if (held_ms < es->min_duration_ms) {
            continue;
        }
        if (es->last_sent_us[c] != 0 && since_ms < es->cooldown_ms) {
            continue;
        }

        send_one(es, (size_t)c < num_labels ? labels[c] : "unknown", best_per_class[c]);
        es->last_sent_us[c] = now;
    }
}
