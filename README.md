# computer-vision  
CV class experiment, when learning at Shandong University   

## Before run code       
The enviroment is visual studio 2022,equipped with opencv c++     

## Context of every code   
2_1  alpha混合的优化   
2_2  对比度调整   
3_1  电子哈哈镜   
4_1  gauss filter  
5_1  bilinear filter  
6    harris corner detect  
7    feature detect&match  (from here,the code is running based on python opencv)

E8 compare SIFT and R2D2  
---
    notice that the code of r2d2 is from [code](http://github.com/naver/r2d2)  
    how to run the r2d2,is like：  
    ```python r2d2/extract.py --model r2d2/models/r2d2_WASF_N16.pt --images imgs/2.jpg --top-k 2056```  
    the eviroment reuqires( my env): python 3.6.8  pytorch 9.0  pillow  8.4.0  matplolib 2.2.2    (the pillow should install by pip,instead of default)  

