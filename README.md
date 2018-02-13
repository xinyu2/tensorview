# tensorview
use paraview to visualize CNN training
## install paraview
1. [official installation](https://www.paraview.org/Wiki/ParaView:Build_And_Install)

2. replace the according path to paraview libraries in the *mcnn_ada2_insitu.py*
```
sys.path.append('/home/chen/software/paraview_build/lib')
sys.path.append('/home/chen/software/paraview_build/lib/site-packages')
```

## Three library files
1. BuildSet.py
find similar filter and put them into set.

2. reNN.py
load and restore weights and biases variable

3. act2grdcp.py
construct the layout of scatter points of activations and output them to paraview catalyst pipeline

## One example to use the above library files in visualizing training of a CNN for the MNIST dataset.
1. mcnn_ada2_insitu.py
```python
python mcnn_ada2_insitu.py --interval=1 --alpha=0.85 --totalepoch=10
```

## use paraview to visualize
1. open paraview GUI
2. select Catalyst->Connect, confirm port *22222*
3. select input from the *Pipiline Browser*
4. click the eye-icon besides *Extract input*
The activations are rendered as images. Use mouse to zoom and rotate for better view. 

