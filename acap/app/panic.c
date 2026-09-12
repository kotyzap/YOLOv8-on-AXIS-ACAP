/**
 * Copyright (C) 2025, Axis Communications AB, Lund, Sweden
 * Modifications Copyright (C) 2026 Pavel Kotyza <kotyza@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "panic.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <syslog.h>
#include <unistd.h>

// runMode is "respawn", so ACAP restarts the app as soon as it exits. A fatal
// condition that persists -- most likely AXIS Object Analytics holding the
// DLPU -- would otherwise become a tight crash loop that fills the system log.
// Back off first so the retry is once every few seconds, not continuous.
#define PANIC_BACKOFF_SECONDS 5

// Function definition for panic
__attribute__((noreturn)) __attribute__((format(printf, 1, 2))) void panic(const char* format,
                                                                           ...) {
    va_list arg;
    va_start(arg, format);
    vsyslog(LOG_ERR, format, arg);
    va_end(arg);
    sleep(PANIC_BACKOFF_SECONDS);
    exit(1);
}
