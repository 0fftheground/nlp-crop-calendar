## 接口文档
更新日期：2026-03-11
基础信息：服务名：Agri Weather APIBase 
URL：http://10.109.2.203:15637
据格式：application/json
统一响应格式{
  "code": 200,
  "message": "说明信息",
  "data": []
}
### 1. 农事气象与适宜度查询
POST /suit_rili
请求体（JSON）{
  "farm_id": 1,
  "region_id": null,
  "start_date": "20260101",
  "end_date": "20260331"
}
字段说明：farm_id：可选，农场ID（二选一）region_id：可选，区域ID（二选一）start_date：必填，开始日期，格式 YYYYMMDDend_date：必填，结束日期，格式 YYYYMMDD
说明：farm_id 与 region_id 必须且只能传一个。传 farm_id：读取 agri_farm 中该农场中心点经纬度，计算适宜度并写入 agri_weather。传 region_id：读取 agri_region 中该区域中心点经纬度，计算适宜度并写入 agri_weather_region。单年区间直接查询对应年份，跨年区间会按年份依次获取后合并。返回结果会按 date 升序排序，并按闭区间 [start_date, end_date] 截取。agri_weather_region 表及相关文档统一使用字段名 region_id。
响应
    成功：code=200，data 为当年每日气象与适宜度列表。 
    失败：code=400 或 500。
    响应示例：{
    "code": 200,
    "message": "成功获取20260101至20260331数据，共90条。",
    "data": [
        {
        "date": "2026-01-01",
        "tmax": 3.43,
        "tmin": -0.48,
        "tavg": 0.77,
        "wins": 2.99,
        "pre": 5.28,
        "rh": 91.29,
        "sf_ws": 0.6,
        "sf_reason": "有小雨，不建议施肥。",
        "lm_ws": 0.0,
        "lm_reason": "温度过低，极不适合炼苗。",
        "yz_ws": 0.0,
        "yz_reason": "近三日气温过低，极不适合移栽。",
        "fd_ws": 0.8,
        "fd_reason": "有小雨，不建议翻地。",
        "dy_ws": 0.0,
        "dy_reason": "温度过低，极不适合打药",
        "sg_ws": 0.0,
        "sg_reason": "有较强降水，极不适合收割。",
        "zd_ws": 0.5,
        "zd_reason": "有小雨，不建议整地。"
        }
    ]
    }
    错误示例：{
    "code": 400,
    "message": "start_date 和 end_date 格式必须为 YYYYMMDD。",
    "data": []
    }
4. 播种适宜期查询POST /bozhong_syd
    请求体（JSON）{
    "farm_id": 1,
    "region_id": null,
    "culti_type": 4,
    "sowing_method": 0,
    "sub_type": 9,
    "crop": 0
    }
    字段说明：farm_id：可选，农场ID（二选一）region_id：可选，区域ID（二选一）culti_type：必填，稻作类型字典 codesowing_method：必填，种植方式参数sub_type：必填，亚种类型字典 codecrop：可选，当前仅支持 0（水稻）说明：farm_id 与 region_id 必须且只能传一个。接口会先从数据库读取目标对象的 province_code，并将该值传给 dateget()。本接口当前不需要请求中直接传经纬度。
    成功响应示例{
    "code": 200,
    "message": "success",
    "data": {
        "suitDate": ["2026-03-25", "2026-03-26"],
        "unsuitDate": [""],
        "unsuitReasons": [""]
        }
    }
    失败响应示例{
    "code": 400,
    "message": "farm_id 和 region_id 必须且只能传一个；缺少 culti_type",
    "data": null
    }