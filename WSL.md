## Windows Subsystem for Linux
### INTRODUCTION
Windows下Linux子系统(Windows Subsystem for Linux，WSL)是一个为在Windows 10上能够原生运行Linux二进制可执行文件的兼容层[1]。该兼容层使得纯正的Ubuntu14.04
能够在Windows环境下原生运行。
### AQUIRMENT
Windows 10 版本：>= 1709  
体系结构：X64  
系统盘可用空间：200M
### INSTALL
step1: 控制面板中选择**程序和功能**  
step2: 勾选**适用于Linux的Windows子系统**并重启  
step3: 在应用商店中查找Ubuntu并安装即可
### SETTING
安装成功后，即可启动。可以选择Ubuntu应用启动，也可在终端中输入Bash命令进入。由于Windows的终端环境配色并不是很友好，可使用[调色工具](https://github.com/Microsoft/console/releases/tag/1708.14008)进行更改。
共有五种配色方案，终端下运行如下命令即可：
<pre><code>
colortool.exe -b deuteranopia
colortool.exe -b OneHalfDark
colortool.exe -b OneHalfLight
colortool.exe -b solarized_dark
colortool.exe -b solarized_light
</code></pre>
同时需要换源，手动对/etc/apt/sources.list文件进行修改。个人喜欢使用科大源：
<pre><code>
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.ustc.edu.cn/ubuntu/ xenial main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ xenial main main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
 
# 预发布软件源，不建议启用
# deb https://mirrors.ustc.edu.cn/ubuntu/ xenial-proposed main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ xenial-proposed main restricted universe multiverse
</code></pre>

### REFERENCE
[1]https://baike.baidu.com/item/wsl/20359185?fr=aladdin
