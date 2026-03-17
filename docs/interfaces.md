# 接口文档汇总

更新日期：2026-03-12

## （一）气象适宜度接口文档

### 基础信息

- 所属项目：`weather_demo`
- 服务名：Agri Weather API
- Base URL：`http://10.109.2.203:15637`
- 数据格式：`application/json`
- 详细文档：`weather_demo/weather_api.md`

### 统一响应格式

```json
{
  "code": 200,
  "message": "说明信息",
  "data": []
}
```

### 接口汇总

#### 1. 农事气象与适宜度查询

- 方法与路径：`POST /suit_rili`
- 用途：按农场或区域查询指定日期区间内的逐日气象与农事适宜度结果
- 关键入参：
  - `farm_id` / `region_id`：二选一
  - `start_date`：`YYYYMMDD`
  - `end_date`：`YYYYMMDD`
- 说明：
  - 单年区间直接按该年处理
  - 跨年区间按年份获取后合并
  - 返回结果按 `date` 升序，并按闭区间截取

请求示例：

```json
{
  "farm_id": 1,
  "region_id": null,
  "start_date": "20260101",
  "end_date": "20260331"
}
```

#### 2. 新增农场

- 方法与路径：`POST /add_farm`
- 用途：新增农场，并可选上传边界文件生成农田地块

#### 3. 删除农场

- 方法与路径：`POST /delete_farm`
- 用途：按农场名称删除农场

#### 4. 播期推荐

- 方法与路径：`POST /bozhong_syd`
- 用途：根据农场或区域、稻作类型、种植方式、亚种类型进行播期推荐
- 关键入参：
  - `farm_id` / `region_id`：二选一
  - `culti_type`
  - `sowing_method`
  - `sub_type`
  - `crop`：当前仅支持 `0`
- 说明：
  - 接口先从数据库读取 `province_code`
  - 不需要请求中直接传经纬度

## （二）动态农事日历接口文档

### 基础信息

- 所属项目：`crop-calender`
- Base URL：`http://10.109.2.203:26405`
- 详细文档：`crop-calender/crop-calender-API.md`

### 响应格式

成功：

```json
{
  "code": "0",
  "msg": "",
  "data": {}
}
```

失败：

```json
{
  "code": "0404400",
  "msg": "错误信息",
  "data": ""
}
```

### 接口汇总

#### 1. 预览农事日历

- 方法与路径：`POST /cropCalender/previewCalender`
- 用途：统一支持农场模式和区域模式的农事推荐预览
- 关键入参：
  - `farm_id` / `region_id`：二选一
  - `sowing_date`
  - `variety_id`
  - `sowing_method`
  - `transp_date`
  - `culti_type`
- 说明：
  - `farm_id` 模式会创建一条未启用的种植计划，并返回 `plant_season_id`
  - `region_id` 模式不创建种植计划
  - 两种模式都只做预览，不落推荐结果
  - 旧接口 `/cropCalender/plantPlan/add` 已废弃

#### 2. 启用种植计划

- 方法与路径：`POST /cropCalender/plantPlan/activate`
- 用途：启用或停用种植计划
- 关键入参：
  - `plant_season_id`
  - `is_active`
- 说明：
  - 当 `is_active=true` 时，会根据该计划重新计算并落库生育期预测和农事推荐

#### 3. 根据种植计划刷新农事日历

- 方法与路径：`POST /cropCalender/refreshCalenderByPlantPlan`
- 用途：按种植计划 ID 读取计划、实际生育期、实际农事记录后重算并落库
- 关键入参：
  - `plant_season_id`

#### 4. 删除种植计划

- 方法与路径：`POST /cropCalender/plantPlan/delete`
- 用途：删除种植计划及其关联预测、推荐、实际数据

## （三）平台前端接口及数据库查询文档

### 基础信息

- 所属项目：`crop-calendar-fe/server`
- 服务端 Base URL：`http://10.109.2.203:35168`
- 本地 Base URL：`http://127.0.0.1:5000`
- 详细文档：`crop-calendar-fe/界面及生育期与农事录入API.md`

### 通用说明

- 涉及种植计划的查询、录入、刷新接口均只面向 `is_active=true` 的计划
- 新增数据库查询接口统一返回：
  - `code=200`：查询成功且有数据
  - `code=204`：查询成功但无数据
  - `code=400`：参数错误

### 前端交互接口

#### 1. 登录

- `POST /api/login`

#### 2. 获取菜单

- `GET /api/menu`
- 用途：获取当前启用计划菜单

#### 3. 实际生育期记录

- `GET /api/stage_dates/{plan_id}`：获取实际生育期记录
- `POST /api/stage_dates/{plan_id}`：新增或更新实际生育期记录
- `DELETE /api/stage_dates/{plan_id}`：删除实际生育期记录

#### 4. 农事记录

- `GET /api/tasks/{plan_id}`：获取农事记录列表
- `POST /api/tasks/{plan_id}`：新增农事记录
- `DELETE /api/tasks/{plan_id}`：删除农事记录

说明：

- 新增农事支持 `is_completed`、`detail`
- 若未传 `is_completed`，按 `work_date` 与当天比较自动判定
- 同一计划下同一农事同一天不可重复添加
- 若农事名称不在该计划推荐农事中，使用字典表 `code=78`（其他农事）

#### 5. 批量刷新计划

- `POST /api/refresh_all`
- 用途：调用动态农事日历服务刷新全部启用计划

### 数据库查询接口

#### 1. 按条件检索种植计划

- `POST /planting-plan/search`

#### 2. 查询启用的种植计划列表

- `GET /planting-plan/active`

#### 3. 查询单个种植计划基础信息

- `GET /planting-plan/{plan_id}`

#### 4. 获取种植计划聚合详情

- `GET /planting-plan/detail/{plan_id}`

#### 5. 按计划 ID 查询最新预测生育期

- `GET /growth-stage/by-plan/{plan_id}`

#### 6. 查询最近一周农事

- `GET /farm-work/recent-week/{farm_id}`
- 说明：
  - 查询农场下启用计划未来 7 天内的农事安排
  - 按计划分组返回农事列表，每条包含 name + date
  - 农场不存在时返回 {"code":204,"msg":"未查询到该农场","data":{"farm_id":...,"plans":[]}}
  - 农场无启用计划时返回 {"code":204,"msg":"未查询到该农场启用的种植计划","data":{"farm_id":...,"plans":[]}}

## 备注

- `weather_demo` 负责气象与播期推荐能力
- `crop-calender` 负责动态农事日历计算与落库
- `crop-calendar-fe/server` 负责平台前端对接与数据库查询
