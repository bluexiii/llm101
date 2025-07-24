# LLM101：从零开始学习大型语言模型

[![Hugo](https://img.shields.io/badge/Hugo-0.148.1+-blue.svg)](https://gohugo.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Website](https://img.shields.io/badge/Website-LLM101-blue.svg)](https://llm101.github.io)

LLM101是一个综合性的学习平台，致力于帮助初学者从零开始掌握大型语言模型的核心概念、应用和最新研究进展。

## 🎯 项目特色

- **结构化学习路径**: 从数学基础到高级应用的完整学习路径
- **高质量内容**: 精心筛选的优质学习资源
- **中文友好**: 优先提供中文教程和文档
- **持续更新**: 及时跟踪最新研究动态
- **开源免费**: 完全开源，免费使用

## 📚 学习内容

### 基础知识
- 数学基础：线性代数、微积分、概率论
- 编程技能：Python、NumPy、Pandas
- 机器学习：监督学习、无监督学习、模型评估
- 深度学习：神经网络、优化算法、框架使用

### 核心概念
- Transformer架构详解
- 注意力机制深入理解
- 预训练与微调技术
- 提示工程实践

### 应用开发
- LLM API使用
- LangChain框架
- RAG系统构建
- 聊天机器人开发

### 最新研究
- 前沿论文解读
- 研究趋势分析
- 技术发展跟踪
- 权威资源推荐

## 🚀 快速开始

### 本地开发

1. **克隆项目**
   ```bash
   git clone https://github.com/llm101/llm101.git
   cd llm101
   ```

2. **安装Hugo**
   ```bash
   # macOS
   brew install hugo
   
   # Windows
   choco install hugo
   
   # Linux
   sudo apt-get install hugo
   ```

3. **启动开发服务器**
   ```bash
   hugo server --buildDrafts --buildFuture
   ```

4. **访问网站**
   打开浏览器访问 http://localhost:1313

### 部署到GitHub Pages

1. **构建静态文件**
   ```bash
   hugo --minify
   ```

2. **推送到GitHub**
   ```bash
   git add .
   git commit -m "Update content"
   git push origin main
   ```

3. **配置GitHub Pages**
   - 进入仓库设置
   - 启用GitHub Pages
   - 选择部署分支（通常是`gh-pages`或`main`）

## 📁 项目结构

```
llm101/
├── content/                 # 网站内容
│   ├── basics/             # 基础知识
│   ├── roadmap/            # 学习路线图
│   ├── research/           # 最新研究
│   ├── resources/          # 精选资源
│   └── about/              # 关于我们
├── themes/                 # Hugo主题
│   └── llm101-theme/      # 自定义主题
├── static/                 # 静态资源
│   ├── css/               # 样式文件
│   ├── js/                # JavaScript文件
│   └── images/            # 图片资源
├── layouts/               # 布局模板
├── data/                  # 数据文件
├── hugo.toml             # Hugo配置文件
└── README.md             # 项目说明
```

## 🛠️ 技术栈

- **静态网站生成器**: [Hugo](https://gohugo.io/)
- **CSS框架**: [Bootstrap 5](https://getbootstrap.com/)
- **图标库**: [Font Awesome](https://fontawesome.com/)
- **代码高亮**: [Prism.js](https://prismjs.com/)
- **托管平台**: [GitHub Pages](https://pages.github.com/)

## 📖 内容贡献

我们欢迎社区贡献内容！

### 贡献指南

1. **Fork项目**
   ```bash
   git clone https://github.com/your-username/llm101.git
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/new-content
   ```

3. **添加内容**
   - 在`content/`目录下创建新的Markdown文件
   - 使用正确的Front Matter格式
   - 遵循内容组织规范

4. **提交更改**
   ```bash
   git add .
   git commit -m "Add new content: [描述]"
   git push origin feature/new-content
   ```

5. **创建Pull Request**

### 内容规范

- **文件命名**: 使用小写字母和连字符
- **Front Matter**: 包含title、description、date等必要字段
- **Markdown格式**: 遵循标准Markdown语法
- **图片资源**: 放在`static/images/`目录下
- **代码示例**: 使用适当的语法高亮

### 内容类型

- **教程**: 详细的技术教程
- **资源**: 优质学习资源推荐
- **研究**: 最新研究动态
- **项目**: 实践项目案例

## 🤝 社区参与

### 讨论交流

- **GitHub Discussions**: [参与讨论](https://github.com/llm101/llm101/discussions)
- **Issues**: [报告问题](https://github.com/llm101/llm101/issues)
- **Pull Requests**: [贡献代码](https://github.com/llm101/llm101/pulls)

### 联系方式

- **邮箱**: contact@llm101.com
- **GitHub**: [@llm101](https://github.com/llm101)
- **Twitter**: [@LLM101_Official](https://twitter.com/LLM101_Official)

## 📄 许可证

本项目采用 [MIT许可证](LICENSE)。

## 🙏 致谢

感谢所有为这个项目做出贡献的社区成员：

- 内容贡献者
- 技术顾问
- 测试用户
- 开源项目维护者

## 📊 项目统计

- **教程数量**: 50+ 篇详细教程
- **资源链接**: 200+ 优质学习资源
- **研究论文**: 100+ 重要论文摘要
- **代码示例**: 100+ 实用代码示例

## 🎉 最新更新

### v1.0.0 (2024-01-01)
- ✅ 完成网站基础架构
- ✅ 发布完整学习路线图
- ✅ 建立内容更新机制
- ✅ 优化用户体验

## 📈 发展计划

### 短期目标 (2024年Q1)
- [ ] 扩展基础知识内容
- [ ] 添加更多代码示例
- [ ] 优化移动端体验
- [ ] 建立用户反馈系统

### 中期目标 (2024年Q2-Q3)
- [ ] 推出在线课程
- [ ] 建立学习社区
- [ ] 开发学习工具
- [ ] 国际化支持

### 长期目标 (2024年Q4)
- [ ] 成为权威LLM学习平台
- [ ] 建立完整的生态系统
- [ ] 推动AI教育发展
- [ ] 贡献开源社区

---

**开始您的LLM学习之旅吧！** 🚀

访问 [LLM101官网](https://llm101.github.io) 开始学习。
