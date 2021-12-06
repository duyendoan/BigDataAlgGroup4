import numpy as np
import hashlib
from itertools import combinations

# define functions to genrate random vectors (which can be treated as the normal vector of a particular hyperplanes)
# for each document, get its dot product with each of the random vector, if the result is >= 0, return 1 as signature
# else return 0 as signature, finally, the length of a sketch for a document = number of random vectors 
def signature_bit(featureMatrix, planes):
    """
    LSH signature generation using random projection
    :parameter featureMatrix: shape(n_doc, n_features)
    :parameter planes: shape(n_features, n_signature)
    return the signature bits (sketch) for all vectors, shape(n_doc, n_signature)"""
    sigs = []
    for i in featureMatrix:
        a = [0]*planes.shape[1]
        for j in range(planes.shape[1]):
            if np.dot(i, planes[:,j]) >= 0:
                a[j] = 1
        sigs.append(a)        
    sigMatrix = np.vstack(sigs) 
    return sigMatrix

def sketch(featureMatrix, n_feature, n_sig):
    """
    Return sketch vector for all documents
    parameter n_feature: 
    parameter featureMatrix: shape is (nDocs, nFeatures)
    parameter n_sig: number of signatures for each document
    """
    ref_planes = np.random.randn(n_sig, n_feature) # randomly generate n_sig vectors, dimesnion=n_feature, values in N(0,1)
    sketches = signature_bit(featureMatrix, ref_planes.T).T # transpose the result to get (n_signature,n_doc) matrix
    return sketches



# find candidate pairs by hashing the siganitures values, 
# if in at least one of the bands, the two documents hash to the same bucket, they form a candidate pair

def LSH(sigMatrix, b, r):
    """
    map similar documents into the same hash bucket
    param signMatrix: signature matrix with shape(n_signature, n_doc)
    param b: the number of bands
    param r: the number of rows in a band
    return the hash bucket: a dictionary, key is hash value, value is column number
    """
    hashBuckets = {}
    begin, end = 0, r
    count = 0
    while end <= sigMatrix.shape[0]:
        count += 1
        for col in range(sigMatrix.shape[1]):
            h_fn = hashlib.md5()
            band = str(sigMatrix[begin: begin + r, col]) + str(count)
            h_fn.update(band.encode())
            tag = h_fn.hexdigest() # use hash value as bucket tag
            
            if tag not in hashBuckets:
                hashBuckets[tag] = [col]
            elif col not in hashBuckets[tag]:
                hashBuckets[tag].append(col)
        begin += r
        end += r
    return hashBuckets

def find_pairs(li):
    """find combination pairs of similar items in the same bucket"""
    pairs = set()
    for i in combinations(li,2):
        pairs.add(i)
    return pairs
