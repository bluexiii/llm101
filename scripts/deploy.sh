#!/bin/bash

# LLM101 部署脚本
# 用于构建和部署网站到GitHub Pages

set -e

echo "🚀 开始部署 LLM101 网站..."

# 检查Hugo是否安装
if ! command -v hugo &> /dev/null; then
    echo "❌ Hugo 未安装，请先安装 Hugo"
    echo "macOS: brew install hugo"
    echo "Windows: choco install hugo"
    echo "Linux: sudo apt-get install hugo"
    exit 1
fi

# 检查Git是否安装
if ! command -v git &> /dev/null; then
    echo "❌ Git 未安装，请先安装 Git"
    exit 1
fi

# 构建网站
echo "📦 构建网站..."
hugo --minify

# 检查构建是否成功
if [ ! -d "public" ]; then
    echo "❌ 构建失败，public 目录不存在"
    exit 1
fi

echo "✅ 网站构建成功！"

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  检测到未提交的更改"
    echo "请先提交更改："
    echo "  git add ."
    echo "  git commit -m 'Update content'"
    echo "  git push origin main"
else
    echo "✅ 所有更改已提交"
fi

# 显示部署信息
echo ""
echo "🎉 部署完成！"
echo ""
echo "📋 部署信息："
echo "  - 网站地址: https://llm101.github.io"
echo "  - 本地预览: http://localhost:1313"
echo "  - 构建目录: ./public"
echo ""
echo "📝 下一步："
echo "  1. 推送到 GitHub: git push origin main"
echo "  2. 检查 GitHub Actions 部署状态"
echo "  3. 等待几分钟后访问网站"
echo ""
echo "🔧 本地开发："
echo "  hugo server --buildDrafts --buildFuture"
echo ""
echo "�� 更多信息请查看 README.md" 