import streamlit.components.v1 as components

def cartoon_html():
    components.html(
        """
        <!DOCTYPE html>
        <html>
            <head> 
       
                <meta charset="utf-8">
                <style>
                    #div1 {
                 
                        height: 250px;
                        line-height: 250px;
                        width: 200px;
                        text-align: center;
                        text-align: left;
                        float: left
                        }
                    #div2 {
                        height: 180px;
                        line-height: 180px;
                        width: 750px;
                        text-align: center;
                        text-align: left;
                        float: left
                        }
                </style>
            </head>
        
            <body>
                <div id="container" style="width:1000px">
                    <div id="div1">
                        <img src="https://img.zcool.cn/community/01ce27596ae76da8012193a3c0e717.gif" width="300px" />
                    </div>
                    <div id="div2" >
                        <font size='10'><b>	&nbspTimeToolv2  </b></font>
                    </div>
                </div>
            </body>
            
        </html>
    """
    )

def linkone():
    components.html(
        """
        <html>
            <head>
                <meta charset="UTF-8">
            </head>
            <body>
                <a href="https://blog.csdn.net/Be_racle/article/details/112600268" target="_blank">我的CSDN博客：移动平均算法</a>
            </body>
        </html>
        """
    )
