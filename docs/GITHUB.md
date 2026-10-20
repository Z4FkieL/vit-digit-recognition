# 上传到 GitHub

仓库名：`vit-digit-recognition`。

项目已经初始化为本地 Git 仓库，默认分支为 `main`。运行代码、训练后的模型、测试夹具和展示截图纳入版本控制；数据集、训练检查点、临时脚本、历史版本和本地验收日志保留在 `work/`，不会上传。

## 创建远程仓库

在 GitHub 创建名为 `vit-digit-recognition` 的空仓库，根据自己的需要选择公开或私有。创建时不额外生成 README、`.gitignore` 或许可证文件，本地已经包含项目说明和忽略规则。

本项目尚未指定开源许可证；需要授权他人复用时，可另行选择并添加 `LICENSE`。

## 首次推送

在项目目录执行，将用户名替换为自己的 GitHub 用户名：

```powershell
git remote add origin "https://github.com/<你的用户名>/vit-digit-recognition.git"
git push -u origin main
```

GitHub 登录凭据由本机 Git 的凭据管理器处理，不要把访问令牌写入代码或仓库地址。

## 更新项目

```powershell
node tools/check-local.js
node tools/verify.js
node tools/verify-flow.js
git add .
git commit -m "Describe the update"
git push
```

如果想让别人直接通过网址使用实验台，可以在仓库创建并推送后配置静态托管。项目入口为根目录 `index.html`，没有构建步骤或服务器端依赖。
