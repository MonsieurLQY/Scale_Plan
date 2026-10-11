# Scale-Plan 与 RoboPARA 2.0

这是高层任务规划的共享执行器对比路线。上游只用 instruction、初始主相机图片和静态技能说明；下游用仿真真值完成物理技能。它独立于 `robopara.evaluation.v1` RGB-only 底层控制评测。

## 启动

下游在 `scaleplan` 分支，使用其已安装 ManiSkill/SAPIEN 的环境：

```bash
cd ../RoboPARA-2.0
conda activate robopara2
python scripts/serve_scaleplan.py --port 8040 --max-steps 10000 --log-dir experiment_logs/scaleplan
```

上游使用 Python 3.10，安装 `requirements-robopara.txt` 中的依赖；无需启动 AI2-THOR，也无需 Fast Downward 或 OpenAI SDK。凭据沿用 `scripts/gemini_client.py`，不将凭据放进项目。

```bash
python scripts/run_robopara.py --server-url http://127.0.0.1:8040 \
  --instruction instruction.txt --goals-config configs/desktop_cleanup.json --seed 0
```

模型、项目、区域和凭据路径分别由 `--model`、`--project`、`--location`、`--credentials` 或原有环境变量配置；`--fallback-model` 沿用现有 404 回退行为。

当前机器也可复用已有环境，不下载依赖：

```bash
PYTHONPATH=/home/lqy/miniconda3/envs/robopara2/lib/python3.10/site-packages \
MPLCONFIGDIR=/tmp/scaleplan-mpl \
/home/lqy/miniconda3/envs/smartllm/bin/python scripts/run_robopara.py
```

## 更换 instruction

instruction 是意图，编号不是强制执行顺序。规划器决定必要动作、手臂、交接方向和并行关系。明确的用户约束仍必须遵守。初版只支持当前场景和现有技能；新增技能需要同步更新 PDDL、提示词和下游执行映射。

改变目标时，同时提供独立评分配置。例如只收好笔：

```bash
python scripts/run_robopara.py \
  --instruction configs/marker_only_instruction.txt \
  --goals-config configs/marker_only.json --seed 0
```

配置使用 `task_id` 和非空 `goals` 列表。可用目标：`marker_in_holder`、`laptop_closed`、`trash_in_bin`、`mouse_on_laptop`、`cup_upright`。reset 后目标冻结。目标配置不会发给规划模型，也不会从模型输出自动生成；请确保配置与 instruction 一致。

子集仍使用原有物理判据及全场景运动稳定性检查，连续满足 0.5 秒才算成功。完整正式评测的默认五项目标不变。交接属于执行手段，不是强制任务成功条件。

## 保留的方法与适配边界

保留原动作图构造、反向搜索、任务分解、分配、摘要、子计划生成、合并及引用绑定。新增视觉输入和独立桌面动作域；生成子计划显式进入原合并阶段。Gemini 替换模型服务，不替换原方法为单次提示规划。

使用 `robot1=left`、`robot2=right`。动作参数顺序：

```text
pick robotN desktop_surface OBJECT
place robotN OBJECT DESTINATION
handover DONOR_ROBOT RECEIVER_ROBOT OBJECT
flap_close robotN laptop
adjust robotN cup upright
```

末阶段必须输出 `0.000: (动作 参数...) [时长]`。确定性转换保留所有时间区间不重叠关系，允许不同手臂、无共享对象的动作重叠。共享 `desktop_surface` 不构成互斥；同手臂、同物体、同目标支撑物互斥。物理执行由完成事件推进，不用估计时长判定成功。

不增加格式纠错调用或执行后重规划。不自动换臂、补交接、串行化冲突计划或追加扶正/放置补救；保留共享物理执行器既有候选及轨迹重试。未知视觉状态不视为已知；操作引用的对象必须在视觉输出中确认为可见。

原 AI2-THOR 入口与默认 OpenAI 调用保留。新入口只关闭自身不使用的 Fast Downward 初始化及生成目录清理。

## 结果与恢复

默认上游产物在 `logs/robopara/<run_id>/`，保存图片、指令、视觉结果、各阶段 prompt/response、模型调用元数据、计划和评分。下游保存 episode 的 `run.json`、`plan.json`、`events.jsonl`、`trace.json`、`checkpoint.json`、`result.json`。

程序退出码 0 表示计划实际产生了仿真控制步，不代表任务成功；以 `result.json` 的 `task_success` 为准。模型或解析失败会 finalize 并保存失败记录。execute 请求断线后查询结果，绝不重复提交执行。

查询：`GET /v1/scaleplan/result?episode_id=...`。服务是单回合串行运行，执行中不会并行处理查询；执行结束或重启恢复后可读取持久化结果。`--timeout` 为单次请求超时，默认 1800 秒。

## 测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_robopara.py
# 下游
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_scaleplan.py tests/test_skill_executor.py tests/test_desktop_cleanup.py
```

### HTTP 契约

所有 POST 请求包含 `protocol_version: "robopara.scaleplan.v1"`；响应包含相同版本。路由前缀 `/v1/scaleplan/`：

- `POST reset`：`run_id/task_id/seed/goals`；返回 `episode_id/task_id/seed/images.main`。`goals` 省略时使用五项完整目标。
- `GET frame?episode_id=...`：返回初始 JPEG。
- `POST execute`：`episode_id/plan/perception_wallclock_s/planning_wallclock_s`；返回 `result`。
- `POST finalize`：`episode_id/reason` 和可选耗时；记录上游失败，返回 `result`。
- `GET result?episode_id=...`：查询已结束回合的 `result`。

`plan` 严格包含 `task_id/nodes/precedence`，节点严格包含 `node_id/skill/agents/args`。单臂节点的 `agents` 为 `["left"]` 或 `["right"]`，交接为 `["left","right"]`；`args` 使用下游原技能字段，交接方向为 `left_to_right` 或 `right_to_left`。未知字段、非法计划或资源冲突均拒绝，执行提交消耗本次尝试。

服务默认回合上限为 10000 步；`--max-steps` 同时设置执行预算和 Gym 环境 TimeLimit，避免 6000 步提前截断。
