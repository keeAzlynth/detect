import network

AP_SSID = "esp32_ap"
AP_PASSWORD = "esp32_1234"


def start_ap():
    ap = network.WLAN(network.AP_IF)
    if not ap.active():
        ap.active(True)
    ap.config(essid=AP_SSID, password=AP_PASSWORD, authmode=network.AUTH_WPA_WPA2_PSK)
    print("AP started")
    print(ap.ifconfig())

start_ap()