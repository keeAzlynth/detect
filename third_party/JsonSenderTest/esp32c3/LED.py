from machine import Pin, Timer


class BaseLED:
    _next_timer_id = 0
    _max_timer_id = 1

    @classmethod
    def _alloc_timer_id(cls):
        if cls._next_timer_id > cls._max_timer_id:
            raise ValueError("No available hardware timers")
        timer_id = cls._next_timer_id
        cls._next_timer_id += 1
        return timer_id

    def __init__(self, pin_num, timer_id=None):
        self._pin = Pin(pin_num, Pin.OUT, value=0)
        if timer_id is None:
            timer_id = self._alloc_timer_id()
        self._timer = Timer(timer_id)

    def on(self):
        self._timer.deinit()
        self._pin.value(1)

    def off(self):
        self._timer.deinit()
        self._pin.value(0)

    def _toggle(self, _):
        self._pin.value(0 if self._pin.value() else 1)


class DataLED(BaseLED):
    def pulse(self, ms=60):
        self._pin.value(1)
        self._timer.deinit()
        self._timer.init(period=ms, mode=Timer.ONE_SHOT, callback=self._off_cb)

    def _off_cb(self, _):
        self._pin.value(0)


class StatusLED(BaseLED):
    def fast_blink(self):
        self._blink(200)

    def slow_blink(self):
        self._blink(1000)

    def _blink(self, period_ms):
        self._timer.deinit()
        self._timer.init(period=period_ms, mode=Timer.PERIODIC, callback=self._toggle)


class LEDController:
    def __init__(self, data_led_pin, status_led_pin):
        self.data = DataLED(data_led_pin, timer_id=0)
        self.status = StatusLED(status_led_pin, timer_id=1)
