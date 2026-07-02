# 会话写入

## 目标

会话写入这一层要解决的问题是：

> **把已经预处理好的本轮输入，正式写进当前会话的内部消息体系中。**

它关心的不是：

- 模型怎么推理
- 工具怎么执行
- query loop 怎么继续

它关心的是：

```text
这一轮整理好的输入，
到底如何变成当前 session 的正式组成部分。
```

所以这一步本质上是在做：

```text
turn-ready 输入包
→ 当前会话中的正式消息写入
```

---

## 结果

这一层完成之后，系统至少会进入下面这种状态：

- 当前 turn 的输入消息已经进入当前 session 的消息集合
- 后续 query loop 不再面对“原始输入”，而是面对“已写入会话的消息状态”
- 当前 `QueryEngine` 已拥有一份更新后的会话级 messages 视图
- 本轮后续所有模型调用、工具结果回注、状态推进，都会建立在这份会话内消息基础上

所以它的结果不是“生成回答”，而是：

> **当前输入已经成为 session transcript / runtime message state 的一部分。**

---

## 过程

结合 Claude Code 当前这套 orchestration 结构来看，会话写入更适合被理解为：

```text
把输入预处理产出的 message bundle
并入当前 QueryEngine 所维护的会话消息集合
```

这一层大致包含下面几类动作。

### 1. 接收预处理后的 message bundle

输入预处理之后，系统拿到的已经不是原始 prompt，而是一组更适合内部处理的结果，例如：

- 主 user message
- attachment messages
- 某些 meta message
- 控制字段（例如是否进入 query）

这说明会话写入的输入不是：

```text
一条字符串
```

而是：

```text
一个已经整理好的消息包
```

所以会话写入做的第一件事是：

```text
把预处理后的消息包当成“本轮正式输入”
```

### 2. 写入当前 session 级消息缓冲

Claude Code 的 `QueryEngine` 持有的是会话级消息集合，可以理解成：

```text
当前 session 的 mutableMessages
```

所以会话写入的核心动作是：

```text
把本轮新消息并入当前 session 的 messages
```

这里的重点不是“保存到磁盘”，而是：

> **让这次 turn 的输入正式进入当前会话运行态。**

也就是说，在这一层之后：

- 这轮输入已经不再只是外部请求参数
- 它已经变成当前 session 内部 message history 的一部分

### 3. 建立后续 query 的上下文基座

Claude Code 后面的 query loop，不是拿“这次用户原始输入”单独推理，而是拿：

```text
当前会话已存在消息
+
本轮新写入消息
+
system prompt / context / memory
```

一起进入后续处理。

所以会话写入还有一个关键意义：

```text
它决定了后续 query 是基于哪份“当前会话消息视图”运行的
```

换句话说：

> **没有完成会话写入，本轮输入就还没有真正进入 Claude Code 的 runtime 世界。**

### 4. 给后续消息回流预留挂载基础

Claude Code 后面会不断产生新的内部消息，例如：

- assistant message
- tool use message
- tool result message
- progress / system message
- compact 边界消息

这些内容最终都不是孤立存在的，而是继续挂在当前 session 的消息链上。

所以会话写入其实还做了一件隐含的事：

```text
为本轮后续所有消息回流建立锚点
```

也就是说，只有当当前 turn 的输入已经正式进入 session message 链，后面的 assistant / tool / result 才有清晰的归属。

### 5. 与“执行前持久化”区分开

这一点很重要。

会话写入和执行前持久化不是一回事。

### 会话写入更偏：

```text
当前运行态中的消息并入
```

### 执行前持久化更偏：

```text
在 durability / transcript / recovery 视角下留下记录
```

所以不要把它理解成“写数据库”或“写磁盘”。

它首先是：

> **写入当前 session 的内部消息状态。**

至于后面要不要进一步持久化，那是下一层的事情。

### 6. 它在主链路中的位置

如果按当前这套主干顺序理解，会话写入更适合放在这里：

```text
Turn 接收
→ 输入预处理
→ 会话写入
→ 执行前持久化
→ 静态上下文构建
→ Runtime 交接
```

这里的作用非常明确：

- `Turn 接收`：接住一次新输入
- `输入预处理`：把输入整理成 turn-ready bundle
- `会话写入`：把这个 bundle 正式并入当前 session
- 后面的所有步骤都在“已写入的会话状态”上继续推进

---

## 可复用建议

如果以后自己做 AI agent / AI application，这一层有几条很值得直接复用。

### 1. 把“输入进入系统”和“消息进入会话”区分开

不要把外部输入和内部会话消息看成一回事。

更清楚的做法是：

- Turn 接收：系统接住一次外部输入
- 输入预处理：整理成 turn-ready bundle
- 会话写入：把 bundle 正式并入当前 session messages

这样边界会非常清晰。

### 2. 会话写入优先面向 runtime state，而不是持久化

先回答：

```text
这轮输入在当前运行态里落到哪里？
```

再回答：

```text
要不要持久化到磁盘 / transcript / store？
```

把这两个问题拆开，架构会稳很多。

### 3. 用 message bundle，而不是单条字符串做写入单位

真实系统里，一次 turn 很可能不止一条纯文本：

- 主消息
- 附件消息
- 元信息消息
- UI 侧注入的辅助消息

所以更好的写入单位是：

```text
一组内部消息对象
```

而不是：

```text
一个最终 prompt 字符串
```

### 4. 会话写入应该成为后续所有回流消息的锚点

如果后面还会产生：

- assistant 输出
- tool 调用
- tool 结果
- progress 事件

那当前 turn 的输入必须先被正式写入会话，否则后续消息链会缺少清晰归属。

### 5. 让 QueryEngine / Session Engine 持有会话级消息视图

Claude Code 这一层很值得学的一点是：

```text
会话引擎持有当前 session 的 message state
```

这样多轮连续性、工具结果回流、resume、compact 才更容易统一处理。

---

## 一句话总结

> **会话写入的本质，不是保存输入，而是把本轮 turn-ready 输入正式纳入当前 session 的内部消息体系。**

