# 会话管理流程图

下面是当前工程里“会话管理”的 3 张拆分图，按真实代码执行顺序整理。

## 1. 总流程

```mermaid
flowchart TD
    A[用户输入 prompt] --> B[RequestRouter.handle]
    B --> C{prompt 是否为空}
    C -- 是 --> C1[返回 none]
    C -- 否 --> D[读取 pending by session_id]

    D --> E{should_resume_pending?}
    E -- 是 --> F{pending.mode == clarification?}
    F -- 是 --> F1[恢复 clarification<br/>继续当前任务 / 开启新任务]
    F -- 否 --> F2[恢复 pending<br/>_resume_pending]
    F1 --> Z[执行并返回]
    F2 --> Z

    E -- 否 --> G{原来是否存在 pending}
    G -- 是 --> G1[删除 stale pending]
    G -- 否 --> H
    G1 --> H[读取 session_context<br/>构建 contextual_candidate]

    H --> I[IntentRouter.plan<br/>生成 standalone_plan]
    I --> J{高置信 short-circuit contextual?}
    J -- 是 --> J1[直接选 contextual plan]
    J -- 否 --> K{需要线程归属澄清?}
    K -- 是 --> K1[创建 clarification pending<br/>回复 继续当前任务/开启新任务]
    K -- 否 --> L[resolve_session_plan<br/>在 standalone/contextual 间决策]

    J1 --> M[执行前 input validation]
    L --> M
    M --> N{tool / workflow / none}
    N -- tool --> O[execute_tool_plan]
    N -- workflow --> P[execute_workflow_plan]
    N -- none --> Q[respond_none 或 fallback]

    O --> R[update_tool_followup_state]
    P --> S[update_workflow_followup_state]
    R --> T[从响应提取 session_context]
    S --> T
    Q --> U[一般不写 session_context]
    T --> V[写入 session_context_store<br/>并更新 last_context]
    V --> Z[返回响应]
```

## 2. Pending 恢复流程

```mermaid
flowchart TD
    A[读取 pending] --> B{mode 是否为 tool/workflow/clarification}
    B -- 否 --> X[不恢复 pending]
    B -- 是 --> C{是否命中中断规则<br/>plant_plan_list_active / plant_plan_delete / memory_clear}
    C -- 是 --> X
    C -- 否 --> D{缺字段里是否包含 variety}
    D -- 是 --> D1[品种名 / 候选召回 特判]
    D -- 否 --> E
    D1 --> E{pending_kind}

    E -- clarification --> F[只接受<br/>继续当前任务 / 开启新任务]
    E -- confirmation --> G[只接受 yes/no]
    E -- strict_options_only --> H[只接受选项命中]
    E -- 其他 --> I[先匹配 options]

    I --> J{是否单字段 typed reply}
    J -- region_id --> J1[region_like]
    J -- variety --> J2[variety_like]
    J -- date --> J3[date_like]
    J -- plan_id --> J4[id_like]
    J -- name --> J5[task_name_like]
    J -- operator --> J6[operator_like]
    J -- work_desc --> J7[work_desc_like]

    J1 --> K
    J2 --> K
    J3 --> K
    J4 --> K
    J5 --> K
    J6 --> K
    J7 --> K

    J -- 无单字段特判 --> K[structured_pending_reply]
    K --> L{命中?}
    L -- 是 --> M[继续原 tool/workflow]
    L -- 否 --> X[不恢复 pending]

    X --> N{router 中原来有 pending?}
    N -- 是 --> O[删除 stale pending<br/>按新问题处理]
    N -- 否 --> P[直接走新问题路由]
```

## 3. Session Context 续接流程

```mermaid
flowchart TD
    A[上轮执行成功后的响应] --> B{是否仍有 missing_fields?}
    B -- 是 --> C[不写 session_context]
    B -- 否 --> D[extract_session_context_from_tool/workflow]

    D --> E{能提取上下文?}
    E -- 否 --> F[不写]
    E -- 是 --> G[写入 tool_contexts/workflow_contexts]
    G --> H[更新 last_context = 最近一次成功上下文]

    H --> I[下一轮用户输入]
    I --> J[build_contextual_candidate]
    J --> K{只看 last_context 对应 adapter}
    K --> L[build_candidate(prompt, context)]

    L --> M{生成 contextual_candidate?}
    M -- 否 --> N[只用 standalone_plan]
    M -- 是 --> O[同时仍会跑 standalone_plan]
    O --> P{candidate.confidence >= 0.85<br/>且 standalone 为 none 或同 action/name}
    P -- 是 --> Q[直接 short-circuit 选 contextual]
    P -- 否 --> R{短句且 standalone/contextual 冲突?}
    R -- 是 --> S[创建 clarification pending]
    R -- 否 --> T[resolve_session_plan]
    T --> U[standalone / contextual 二选一]
```

## 4. 关键代码入口

- 总入口：[router.py](/f:/workspace/nlp-crop-calendar/src/agent/router.py)
- Pending 判断：[pending_manager.py](/f:/workspace/nlp-crop-calendar/src/agent/pending_manager.py)
- Session context 适配与续接：[session_context.py](/f:/workspace/nlp-crop-calendar/src/agent/session_context.py)
- Standalone 路由：[intent_router.py](/f:/workspace/nlp-crop-calendar/src/agent/intent_router.py)
- 执行与 follow-up 状态写回：[plan_executor.py](/f:/workspace/nlp-crop-calendar/src/agent/plan_executor.py)
