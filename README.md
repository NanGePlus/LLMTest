# 0、本项目相关视频链接          
提供一种LLM集成解决方案，一份代码支持快速同时支持gpt大模型、国产大模型(通义千问、文心一言、百度千帆、讯飞星火等)、本地开源大模型(Ollama)             
https://www.bilibili.com/video/BV12PCmYZEDt/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                
https://youtu.be/CgZsdK43tcY              

# 1、前期准备工作
## 1.1 LangChain简介
LangChain是一个用于开发由大型语言模型(LLM)驱动的应用程序的框架            
官方网址:https://python.langchain.com/v0.2/docs/introduction/             

## 1.2 开发环境搭建 anaconda、pycharm 安装   
anaconda:提供python虚拟环境，官网下载对应系统版本的安装包安装即可           
pycharm:提供集成开发环境，官网下载社区版本安装包安装即可            
详细安装操作视频如下：        
《AGI入门基础集成开发环境搭建Anaconda+PyCharm》              
https://www.bilibili.com/video/BV1tQWje1ErT/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                           
https://youtu.be/myVgyitFzrA               

## 1.3 gpt大模型使用方案            
国内无法直接访问，可以使用代理的方式，具体代理方案自己选择      
这里我自己使用的是云雾API:https://api.wlai.vip/             

## 1.4 非gpt大模型方案 OneAPI安装、部署、创建渠道和令牌 
### （1）OneAPI是什么
官方介绍：是OpenAI接口的管理、分发系统             
支持 Azure、Anthropic Claude、Google PaLM 2 & Gemini、智谱 ChatGLM、百度文心一言、讯飞星火认知、阿里通义千问、360 智脑以及腾讯混元             
### (2)安装、部署
使用官方提供的release软件包进行安装部署 ，详情参考如下链接中的手动部署：                  
https://github.com/songquanpeng/one-api                  
下载OneAPI可执行文件one-api并上传到服务器中然后，执行如下命令后台运行             
sudo chmod -R 777 one-api                   
nohup ./one-api --port 3000 --log-dir ./logs > output.log 2>&1 &            
ps aux | grep one-api              
运行成功后，浏览器打开如下地址进入one-api页面，默认账号密码为：root 123456                 
http://IP:3000/              
### (3)创建渠道和令牌
创建渠道：大模型类型(通义千问)、APIKey(通义千问申请的真实有效的APIKey)             
创建令牌：创建OneAPI的APIKey，后续代码中直接调用此APIKey              

## 1.5 本地开源大模型方案 Ollama安装、启动、下载大模型          
### （1）Ollama是什么
Ollama是一个轻量级、跨平台的工具和库，专门为本地大语言模型(LLM)的部署和运行提供支持          
它旨在简化在本地环境中运行大模型的过程，不需要依赖云服务或外部API，使用户能够更好地掌控和使用大型模型                
### （2）Ollama安装、启动、下载大模型
安装Ollama，进入官网https://ollama.com下载对应系统版本直接安装即可                                                   
启动Ollama，安装所需要使用的本地模型，执行指令进行安装:                                                          
ollama pull qwen2.5:latest    
或       
ollama pull qwen2.5:7b                                                                
其中:                            
qwen2.5对应版本有0.5b、1.5b、3b、7b(latest)、14b、32b、72b                    
执行执行运行大模型进行交互        
ollama run qwen2.5:latest            
或         
ollama run qwen2.5:7b             


# 2、项目初始化
## 2.1 下载源码
GitHub中下载工程文件到本地，下载地址如下：                
https://github.com/NanGePlus/LLMTest             

## 2.2 构建项目
使用pycharm构建一个项目，为项目配置虚拟python环境               
项目名称:LLMTest            
虚拟环境名称:LLMTest               

## 2.3 将相关代码拷贝到项目工程中           
直接将下载的文件夹中的文件llmBasicTest工程文件夹拷贝到新建的项目根目录                        

## 2.4 安装项目依赖          
pip install -r requirements.txt            
每个软件包后面都指定了本次视频测试中固定的版本号     
**注意:** 截止2024.10.18，langchain最新版本为0.3.3,langchain-openai最新版本为0.2.2，建议先使用要求的对应版本进行本项目测试，避免因版本升级造成的代码不兼容。测试通过后，可进行升级测试。                      


# 3、项目测试          
## 3.1 根据自己实际情况调整utils/myLLM.py内容
**openai模型相关配置 根据自己的实际情况进行调整**                   
OPENAI_API_BASE = "https://api.wlai.vip/v1"            
OPENAI_CHAT_API_KEY = "sk-XmrIEFplNArLlYa0E8C5A7C5F82041FdBd923e9d115746D0"          
OPENAI_CHAT_MODEL = "gpt-4o-mini"           
**非gpt大模型相关配置(oneapi方案 通义千问为例) 根据自己的实际情况进行调整**              
ONEAPI_API_BASE = "http://139.224.72.218:3000/v1"            
ONEAPI_CHAT_API_KEY = "sk-0FxX9ncd0yXjTQF877Cc9dB6B2F44aD08d62805715821b85"               
ONEAPI_CHAT_MODEL = "qwen-max"               
**本地大模型相关配置(Ollama方案 llama3.1:latest为例) 根据自己的实际情况进行调整**             
OLLAMA_API_BASE = "http://localhost:11434/v1"                
OLLAMA_CHAT_API_KEY = "ollama"          
OLLAMA_CHAT_MODEL = "qwen2.5:latest"   

## 3.2 调整llmTest.py内容 
**调整:选择使用哪种模型标志设置:**                       
LLM_TYPE = "oneapi"  # openai:调用gpt模型；oneapi:调用oneapi方案支持的模型(这里调用通义千问)                     

## 3.3 运行llmTest.py测试
使用python llmTest.py命令启动脚本                              
