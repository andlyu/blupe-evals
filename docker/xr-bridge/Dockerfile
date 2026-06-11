# XR input appliance: XRoboToolkit PC Service + xrobotoolkit_sdk + the input bridge.
# arm64 Ubuntu 22.04 = same arch+OS as the Jetson Orin, so this replays the proven
# docs/ORIN-SETUP.md recipe (prebuilt headless .deb + pybind built against its .so).
# Runs natively on Apple Silicon. Build:  docker build -t xr-bridge .
# Run:  docker run --rm -p 63901:63901 -p 8765:8765 xr-bridge
FROM arm64v8/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git python3 python3-pip python3-dev cmake g++ make \
    libdouble-conversion3 libicu70 libglib2.0-0 libfontconfig1 libfreetype6 \
    libxkbcommon0 libdbus-1-3 libpcre2-16-0 libb2-1 \
    && rm -rf /var/lib/apt/lists/*

# The exact deb the Orin runs (headless arm64, v1.0.0). dpkg-deb -x avoids postinst scripts.
ARG DEB_URL=https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb
RUN curl -fsSL -o /tmp/service.deb "$DEB_URL" \
    && dpkg-deb -x /tmp/service.deb / && rm /tmp/service.deb \
    && ls /opt/apps/roboticsservice/SDK/arm64/libPXREARobotSDK.so

ENV APP=/opt/apps/roboticsservice \
    LD_LIBRARY_PATH=/opt/apps/roboticsservice:/opt/apps/roboticsservice/lib:/opt/apps/roboticsservice/SDK/arm64 \
    QT_PLUGIN_PATH=/opt/apps/roboticsservice/plugins/ \
    QT_QPA_PLATFORM=offscreen

# Pybind SDK: build only the wrapper, against the deb's prebuilt .so (headers from the
# orin branch — the same pairing the Orin install used; no from-source SDK build).
RUN pip3 install --no-cache-dir pybind11 \
    && git clone --depth 1 https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind /opt/pybind \
    && git clone --depth 1 -b orin https://github.com/XR-Robotics/XRoboToolkit-PC-Service /tmp/svc-src \
    && mkdir -p /opt/pybind/include/aarch64 /opt/pybind/lib/aarch64 \
    && cp /tmp/svc-src/RoboticsService/PXREARobotSDK/PXREARobotSDK.h /opt/pybind/include/aarch64/ \
    && cp -r /tmp/svc-src/RoboticsService/PXREARobotSDK/nlohmann /opt/pybind/include/aarch64/nlohmann \
    && cp /opt/apps/roboticsservice/SDK/arm64/libPXREARobotSDK.so /opt/pybind/lib/aarch64/ \
    && cd /opt/pybind \
    && CMAKE_PREFIX_PATH="$(python3 -m pybind11 --cmakedir)" pip3 install --no-cache-dir -v . \
    && rm -rf /tmp/svc-src \
    && timeout 30 python3 -c "import xrobotoolkit_sdk; print('xrobotoolkit_sdk import OK')"

COPY bridge.py /bridge.py
COPY entrypoint.sh /entrypoint.sh

# 63901 = Quest -> PC Service; 8765 = bridge -> host consumers (docs/XR-INPUT-BRIDGE.md)
EXPOSE 63901 8765
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
