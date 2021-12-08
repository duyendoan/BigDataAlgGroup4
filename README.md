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

### 4. setup of the folders for scripts and data  
![image](https://github.com/duyendoan/BigDataAlgGroup4/blob/main/files/folders_setup.png)  
scripts should be under the `code` folder  
datasets should be under the `data` folder  
models should be stored under the `model` folder

### 5. to run content-based filtering, call **content_based_gc.py** and specify parameters:   -tl, -lsh, -st e.g.    
-tl: title of the query movie, str  
-lsh: whether to use LSH to put similar movies in same bucket first, str, `y` or `n`   
-st: if use LSH, the top movies should be sort on popularity `pop` or cosine similarity `cosine`  
     
```  
python content_based_gc.py -tl MovieTitle -lsh n   
python content_based_gc.py -tl MovieTitle -lsh y -st cosine   
```  

### 6. to run collaborative filtering, call **collab_model_SVD_gc.py** and specify parameters: -uid, -iid, r_ui, -fn. e.g.   
-uid: user ID, interger  
-iid: movie ID, interger (optional)
-r_ui: real rating for the given user-movie pair, float, (optional)
-fn: filename to save the trained svd model or to import the existing svd model , str
```  
python collab_model_SVD_gc.py -uid 1 -iid 31 r_ui 2.5 -fn model_filename  
python collab_model_SVD_gc.py -uid 1 -fn model_filename  
```  

