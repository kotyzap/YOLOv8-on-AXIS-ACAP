#!/bin/sh
# Verification pass for YOLOv8 Detector on an ARTPEC-8 camera.
#
#   CAM=<camera-ip> AUTH=<user>:<password> sh verify/verify-0.9.3.sh
#
# Checks the things that cannot be checked off the camera: that live.json is
# only written while the settings page is watching, that the app is quiet at
# LOG_INFO, and the running version and settings.
set -u
CAM=${CAM:-}
AUTH=${AUTH:-}
if [ -z "$CAM" ] || [ -z "$AUTH" ]; then
    echo "usage: CAM=<camera-ip> AUTH=<user:password> sh $0" >&2
    exit 2
fi
Q="curl -s --digest -u $AUTH"

echo "=== 1. installed and running ==="
$Q "http://$CAM/axis-cgi/applications/list.cgi" | tr '>' '>\n' | grep -i yolov8

echo
echo "=== 2. settings as the app sees them ==="
$Q "http://$CAM/axis-cgi/param.cgi?action=list&group=root.yolov8_detector"

echo
echo "=== 3. arm the live view, then sample it twice 3 s apart ==="
$Q "http://$CAM/axis-cgi/param.cgi?action=update&yolov8_detector.LiveView=$(date +%s)" ; echo
sleep 2
echo "--- sample A ---"; $Q "http://$CAM/local/yolov8_detector/live.json"; echo
sleep 3
echo "--- sample B ---"; $Q "http://$CAM/local/yolov8_detector/live.json"; echo

echo
echo "=== 4. disarm, then confirm live.json stops changing ==="
$Q "http://$CAM/axis-cgi/param.cgi?action=update&yolov8_detector.LiveView=0" ; echo
sleep 2
C1=$($Q "http://$CAM/local/yolov8_detector/live.json")
sleep 5
C2=$($Q "http://$CAM/local/yolov8_detector/live.json")
[ "$C1" = "$C2" ] && echo "unchanged over 5 s -- flash writes are gated" \
                  || echo "STILL CHANGING -- the gate is not working"

echo
echo "=== 5. log volume at LOG_INFO over 20 s (expect a handful, not hundreds) ==="
$Q "http://$CAM/axis-cgi/admin/systemlog.cgi" | grep -c yolov8_detector
sleep 20
$Q "http://$CAM/axis-cgi/admin/systemlog.cgi" | grep -c yolov8_detector

echo
echo "=== 6. last 15 app log lines ==="
$Q "http://$CAM/axis-cgi/admin/systemlog.cgi" | grep yolov8_detector | tail -15
