# BigDataAlgGroup4
## Movie recommendation engine using the content-based filtering and collaborative filtering 
### 1. install anaconda 3.X 
### 2. create virtual environment   

```  
conda create --name myenv
conda activate myenv  
```  

### 3. install reqiured packages listed in the code/requirements.txt   

```  
conda install --file requirements.txt  
```  

### 4. to run content-based filtering, call **content_based_gc.py** and specify parameters: -tl, -lsh, -st e.g.   
     
```  
python content_based_gc.py -tl MovieTitle -lsh n   
python content_based_gc.py -tl MovieTitle -lsh y -st cosine   
```  

### 5. to run collaborative filtering, call **collab_model_SVD_gc.py** and specify parameters: -uid, -iid, r_ui, -fn. e.g.   

```  
python collab_model_SVD_gc.py -uid 1 -iid 31 r_ui 2.5 -fn model_filename  
python collab_model_SVD_gc.py -uid 1 -fn model_filename  
```  

