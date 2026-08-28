import uasyncio as asyncio
import time
import json

from LED import LEDController
from cmd_handler import CommandHandler

HOST = "0.0.0.0"
PORT = 9527

leds = LEDController(data_led_pin=12, status_led_pin=13)
leds.status.fast_blink()
leds.data.off()

# ----------------- 定义对应命令的回调函数 -----------------
def on_init(data):
    print("--> 收到 init 命令:", data)
    # 处理初始化逻辑...
    return {"status": "ok", "ack": "init", "device": data.get("device")}

def on_detect(data):
    print("--> 收到 detect 命令:", data)
    objects = data.get("objects", [])
    # 处理视觉检测到的对象逻辑...
    return {"status": "ok", "ack": "detect", "found_count": len(objects)}

def on_shutdown(data):
    print("--> 收到 shutdown 命令:", data)
    # 处理关机或待机逻辑...
    return None  # 返回 None 默认会发回通用成功信息

# ----------------- 注册命令 -----------------
#
# 已注册命令的 JSON 发送示例（每条消息以换行符 \n 结尾）：
#
#   {"cmd": "init",    "data": {"device": "esp32c3"}}
#   {"cmd": "detect",  "data": {"objects": ["cat", "dog"]}}
#   {"cmd": "shutdown","data": {}}
#
cmd_parser = CommandHandler()
cmd_parser.register("init", on_init)
cmd_parser.register("detect", on_detect)
cmd_parser.register("shutdown", on_shutdown)


async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print("Client connected:", addr)
    leds.status.on()
    leds.data.off()
    
    try:
        while True:
            # Use readline so we get complete JSON strings (delimited by \n)
            data = await reader.readline()
            if not data:
                print("Client disconnected (empty recv)")
                break
            
            leds.data.pulse(80)
            
            # 打印收到的原始字节（用于调试）
            print(f"[RX] raw bytes ({len(data)}B): {data}")
            
            # 读取字节并解码为字符串
            text = data.decode('utf-8').strip()
            print(f"[RX] decoded text: {text}")
            
            # 使用独立模块的逻辑去解析并触发对应回调函数
            response_dict = cmd_parser.handle_message(text)
            
            # 打印将要返回的响应内容
            print(f"[TX] response: {response_dict}")
            
            # 将处理结果序列化为 JSON 发回给客户端
            resp = json.dumps(response_dict) + "\n"
            writer.write(resp.encode('utf-8'))
                
            await writer.drain()
            leds.data.pulse(80)
    except Exception as e:
        print("Client error:", e)
    finally:
        print("Closing client connection...")
        writer.close()
        await writer.wait_closed()
        leds.data.off()
        leds.status.fast_blink()
        print("Waiting for new connection...")


async def main():
    print(f"TCP server listening on {HOST}:{PORT}")
    server = await asyncio.start_server(handle_client, HOST, PORT)
    while True:
        await asyncio.sleep(3600)

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Server stopped")
