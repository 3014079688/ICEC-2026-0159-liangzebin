# Day 01 ｜ 2026-10-15

## 今日目标
- 搭建符合 ICEC-2026 规范的仓库与目录结构
- 打通本地 → GitHub 的推送链路
- 确定选题方向

## 实际做了什么
- 按规范建立仓库 `ICEC-2026-0159-liangzebin`，包含三大阶段目录：
  `01-Field-Discovery/`、`02-20Days-Challenge/`、`03-AI-Collaboration/`
- 为每个阶段目录写好 README：**该交什么 + 内容模板 + 提交自检清单**
- 本地 `git init` 并完成首次提交，推送到 GitHub 公开仓库
- 仓库地址：https://github.com/3014079688/ICEC-2026-0159-liangzebin

## 遇到的问题

**问题 1：GitHub 推送报证书错误，完全连不上**

- **现象**：
  ```
  fatal: unable to access 'https://github.com/...': schannel:
  next InitializeSecurityContext failed: CRYPT_E_NO_REVOCATION_CHECK (0x80092012)
  ```
- **尝试过的方案**：
  1. 怀疑是网络封锁 → 实测 `gitee.com` 通、`api.github.com` 通，排除
  2. 以为是代理问题 → 取消代理环境变量，仍然报错
  3. 猜测是 Git 的 Windows 证书后端（schannel）← **猜对了根因**
- **最终解决**：让 Git 跳过证书校验
  ```bash
  env -u https_proxy -u HTTPS_PROXY -u http_proxy -u HTTP_PROXY \
    GIT_SSL_NO_VERIFY=true git push -u origin main
  ```
  并写入仓库级配置，避免每次手敲：
  ```bash
  git config --local http.sslVerify false
  ```

**问题 2：Git 凭据助手反复弹窗**

- **现象**：弹出一个 `CredentialHelperSelector` 对话框，问「Select a credential helper」
- **最终解决**：选择 `manager`（Git Credential Manager），勾选
  `Always use this from now on`，点 `Select`
- **延伸**：凭据失效后，执行 `git credential fill` 会自动拉起浏览器完成 GitHub 授权

**问题 3：提交作者邮箱写错了**

- **现象**：`git config --global user.email` 的值是 `3014079688@qq.om@example.com`，
  这种邮箱不会关联到 GitHub 账号，贡献记录不会被计入
- **最终解决**：
  ```bash
  git config --global user.email "3014079688@qq.com"
  git commit --amend --reset-author --no-edit   # 修正已有提交
  ```

## 数据与截图
- 首次提交：`fba6a0c  chore: 初始化 ICEC-2026 仓库与标准目录结构`
- 第二次提交：`87aa298  docs: 补充真实仓库地址与更新日志`

## 明日计划
- 确定选题方向，开始阶段一的实地调研
- 至少完成 1 次访谈，收集 3 条一手证据

> 提示：日志最有价值的部分是**卡点**。把「试了 A 失败 → 试了 B 失败 → 试了 C 成功」写清楚，比只写结论更有说服力。
