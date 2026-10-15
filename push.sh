#!/usr/bin/env bash
# 一键提交并推送
# 用法: bash push.sh "docs: Day02 完成访谈记录"
set -e
cd "$(dirname "$0")"

MSG="${1:-chore: 日常更新}"

git add .
if git diff --cached --quiet; then
  echo "没有需要提交的改动。"
else
  git commit -m "$MSG"
fi

# 取消代理变量，避免 CONNECT tunnel failed 502
env -u https_proxy -u HTTPS_PROXY -u http_proxy -u HTTP_PROXY git push

echo ""
echo "✅ 已推送到 GitHub: https://github.com/3014079688/ICEC-2026-0159-liangzebin"
