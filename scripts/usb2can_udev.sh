#!/bin/bash

# === Configuration ===
VENDOR_ID="1d50"
PRODUCT_ID="606f"
BITRATE="1000000"

# === Script paths ===
UDEV_RULES_PATH="/etc/udev/rules.d/90-canusb.rules"
CAN_UP_SCRIPT="/usr/local/bin/can0-up.sh"
CAN_DOWN_SCRIPT="/usr/local/bin/can0-down.sh"

# === Create can0-up.sh ===
echo "Creating $CAN_UP_SCRIPT..."
sudo tee "$CAN_UP_SCRIPT" > /dev/null <<EOF
#!/bin/bash
sleep 1  # give interface time to appear
/usr/sbin/ip link set can0 type can bitrate $BITRATE
/usr/sbin/ip link set up can0
EOF

sudo chmod +x "$CAN_UP_SCRIPT"

# === Create can0-down.sh ===
echo "Creating $CAN_DOWN_SCRIPT..."
sudo tee "$CAN_DOWN_SCRIPT" > /dev/null <<EOF
#!/bin/bash
/usr/sbin/ip link set down can0
EOF

sudo chmod +x "$CAN_DOWN_SCRIPT"

# === Create udev rule ===
echo "Creating udev rule at $UDEV_RULES_PATH..."
sudo tee "$UDEV_RULES_PATH" > /dev/null <<EOF
ACTION=="add", SUBSYSTEM=="net", KERNEL=="can0", ATTRS{idVendor}=="$VENDOR_ID", ATTRS{idProduct}=="$PRODUCT_ID", RUN+="$CAN_UP_SCRIPT"
ACTION=="remove", SUBSYSTEM=="net", KERNEL=="can0", RUN+="$CAN_DOWN_SCRIPT"
EOF

# === Reload udev ===
echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Done! Plug in your USB2CAN module to test."
