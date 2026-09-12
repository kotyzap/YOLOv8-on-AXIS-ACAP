PROG1	= yolov8_detector
OBJS1	= $(PROG1).c argparse.c events.c channel_util.c img_util.c labelparse.c model.c model_preprocessing.c panic.c
PROGS	= $(PROG1)
DEBUG_DIR = debug

PKGS = axevent axparameter bbox gio-2.0 gio-unix-2.0 liblarod vdostream

CFLAGS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags $(PKGS))
LDLIBS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs $(PKGS))
LDLIBS += -lm
CFLAGS  += -pthread

APP_VERSION = $(shell sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' manifest.json | head -1)
CFLAGS += -DAPP_VERSION=\"$(APP_VERSION)\"

CFLAGS += -Wall -Wextra -Wformat=2 -Wpointer-arith -Wbad-function-cast \
          -Wstrict-prototypes -Wmissing-prototypes -Wdisabled-optimization \
          -Wfloat-equal -W -Werror

all:	$(PROGS)

$(PROG1): $(OBJS1)
	install -d $(DEBUG_DIR)
	$(CC) $^ $(CFLAGS) $(LDFLAGS) $(LDLIBS) -o $(DEBUG_DIR)/$@
	cp $(DEBUG_DIR)/$@ .
	$(STRIP) $@

clean:
	rm -rf $(PROGS) *.o *.eap* *_LICENSE.txt package.conf* param.conf tmp* $(DEBUG_DIR)
