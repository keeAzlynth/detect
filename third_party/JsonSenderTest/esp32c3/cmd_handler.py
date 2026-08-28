import json

class CommandHandler:
    def __init__(self):
        self._callbacks = {}

    def register(self, command_name, callback):
        """
        注册一个命令以及它对应的回调函数。
        回调函数会接收一个参数：解析后的完整 JSON 字典。
        """
        self._callbacks[command_name] = callback

    def handle_message(self, text):
        """
        解析传入的字符串，如果符合命令列表内容则执行并返回结果。
        """
        text = text.strip()
        if not text:
            return {"status": "ignored", "reason": "empty"}
            
        try:
            msg = json.loads(text)
        except ValueError:
            return {"status": "error", "reason": "invalid_json"}
        
        # 确保传入的是字典结构
        if type(msg) is not dict:
            return {"status": "error", "reason": "not_a_dict"}

        # 提取命令字段
        cmd = msg.get("command")
        if not cmd:
            return {"status": "error", "reason": "missing_command"}

        # 若命令被注册过，则调用对应的回调函数
        if cmd in self._callbacks:
            try:
                # 执行回调函数，传入完整的字典数据
                result = self._callbacks[cmd](msg)
                
                # 如果用户回调没有专门返回可用于返回的字典，使用默认成功状态
                if result is None:
                    return {"status": "ok", "ack": cmd}
                return result
            except Exception as e:
                return {"status": "error", "reason": "callback_exception", "detail": str(e)}
        else:
            # 命令未识别
            return {"status": "error", "reason": "unknown_command"}
