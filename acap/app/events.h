/**
 * Custom camera event for YOLOv8 detections.
 *
 * Declares one stateless event on the camera's event system. Any VMS that
 * already consumes Axis events (AXIS Camera Station, Genetec, Milestone) can
 * subscribe to it with nothing extra deployed.
 *
 * Topic: tnsaxis:CameraApplicationPlatform/tnsaxis:YOLOv8Detector/tnsaxis:Detection
 * Data:  class (string), confidence (double)
 */
#pragma once

#include <glib.h>
#include <stddef.h>

typedef struct event_sender event_sender_t;

/**
 * Declares the event. Returns NULL if the event system refuses the
 * declaration; the caller should carry on without events rather than fail.
 *
 * min_duration_ms: how long a class must stay above the confidence threshold
 *                  before it is worth reporting.
 * cooldown_ms:     minimum gap between two events for the same class.
 */
event_sender_t*
event_sender_new(int num_classes, int min_duration_ms, int cooldown_ms);

void event_sender_free(event_sender_t* es);

/** Changes the timing while running. Pass -1 to leave a value alone. */
void event_sender_set_timing(event_sender_t* es, int min_duration_ms, int cooldown_ms);

/**
 * Call once per frame. best_per_class[c] is the highest confidence seen for
 * class c in this frame, or 0 when the class is absent.
 */
void event_sender_update(event_sender_t* es,
                         const float* best_per_class,
                         char** labels,
                         size_t num_labels);
