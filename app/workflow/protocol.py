"""受限的界面选项协议：建议不等于执行，标记不展示给用户。"""

OPTIONS = {"handoff": "转人工", "create_ticket": "建工单"}


class OptionStream:
    """增量剥离控制标记；普通文字立即返回，不缓存整篇回答。"""

    PREFIX = "[[option:"

    def __init__(self):
        """保存跨片段的标记残片与已识别建议；每次模型输出新建一个解析器。"""
        self.buffer = ""
        self.actions = []

    def feed(self, text):
        """消费一个文本片段，返回此刻可以展示的正文，建议另存在actions中。"""
        self.buffer += text
        visible = ""
        while self.buffer:
            start = self.buffer.find(self.PREFIX)
            if start >= 0:
                # 完整前缀之前的正文可立即交出，标记从buffer开头单独处理。
                visible += self.buffer[:start]
                self.buffer = self.buffer[start:]
                end = self.buffer.find("]]")
                if end < 0:
                    # 例如先收到[[option:，后收到create_ticket]]；先留残片等下次。
                    # 长度限制避免模型一直不闭合标记导致无限缓存。
                    if len(self.buffer) > 128:
                        raise ValueError("Invalid option marker")
                    break
                action = self.buffer[len(self.PREFIX) : end]
                # 未知建议不渲染；已出现过的建议只保留一次，不能生成任意按钮动作。
                if action in OPTIONS and action not in self.actions:
                    self.actions.append(action)
                self.buffer = self.buffer[end + 2 :]
                continue
            # 只保留可能跨网络片段的前缀，其余文字可以马上输出。
            keep = 0
            for size in range(1, len(self.PREFIX)):
                if self.buffer.endswith(self.PREFIX[:size]):
                    keep = size
            visible += self.buffer[:-keep] if keep else self.buffer
            self.buffer = self.buffer[-keep:] if keep else ""
            break
        return visible

    def finish(self):
        """流结束时交出普通尾文，丢弃未闭合的控制标记，不补造一个可点击动作。"""
        tail = "" if self.buffer.startswith("[[") else self.buffer
        self.buffer = ""
        return tail
