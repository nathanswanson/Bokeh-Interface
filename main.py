#dale9009@stthomas.edu
#nadi9686@stthomas.edu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from bokeh.plotting import figure, output_file, show
from bokeh.models import FileInput, Button
from bokeh.layouts import column, row, gridplot, layout
from bokeh.models.widgets import Div
from bokeh.io import curdoc, push_notebook
from bokeh.palettes import Category20c
from bokeh.transform import cumsum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from PIL import Image

from ipywidgets import FloatSlider, interact

from functools import partial

epoch = 3
learning_rate = 0.01
momentum = 0.5
log = 10

#
# Defined class Net representing the structure and flow of the network
#
#

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 27)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
# Grabs Datasets for testing and training, downloads them if not present
#
#

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.EMNIST('/files/', train=True, download=True, split='letters',
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.EMNIST('/files/', train=False, download=True, split='letters',
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1000, shuffle=True)

#
# Testing if the file grab works by displaying images
#
#


#
# Initialize network and train
#
#

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(epoch + 1)]


def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
  
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'C:/Users/minne/network/results/model.pth')
      torch.save(optimizer.state_dict(), 'C:/Users/minne/network/results/optimizer.pth')
        
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
     # x = torchvision.transforms.Normalize(mean= 0, std= 0.1)
      #x(target)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

#END OF CALCULATIONS START OF GUI


pred = Div(text="<style>p {font-family:'Lucida Console', Courier, monospace;font-size:24px}</style><p>Prediction: </p>")
actual = Div(text = "<style>p {font-family:'Lucida Console', Courier, monospace;font-size:24px}</style><p>Actual: </p>")
div_image = Div(text="""<img src="network/static/loading_pic.gif" alt="div_image">""", width=100, height=100)

#statistics
correct= 0
index = 0

letters = 26
iscorrect = 2 
correctByLetter = [ [ 0 for i in range(letters) ] for j in range(iscorrect) ]


def eval():
    for data, target in test_loader:
        global index
        global correct
        global bar_graph
        global correctByLetter

        output = network(data)
        predi = output.data.max(1, keepdim=True)[1]
        
        #update preview image 
        plt.imshow(data[index].squeeze(), cmap='gray_r')
        plt.savefig('C:/Users/minne/network/static/current{}.png'.format(index), bbox_inches='tight', dpi=50)
        div_image.text = """<img src="network/static/current{}.png" alt="div_image">""".format(index)
        targ = chr(target[index] + 64)
        prediction = chr(predi[index] + 64)
        if(prediction == targ):
            correct+=1
            correctByLetter[0][target[index] - 1] += 1
        else:
            correctByLetter[1][target[index] - 1] += 1

        #update correct graph
        cat = ["correct", "failed"]
        count = [correct, index-correct]
        bar_graph.vbar(x=cat, top=count, width=0.9)

        #update letter correction graph
        colors = ["#718dbf", "#e84d60"]
        letter_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        data = {'letter_list' : letter_list,
        'correct'   : correctByLetter[0],
        'failed'   : correctByLetter[1]
        }
  
        letter_bar_graph.vbar_stack(cat,x='letter_list', width=0.4, source=data, color=colors)
        letter_bar_graph.y_range.start = 0
        letter_bar_graph.x_range.range_padding = 0.1
        letter_bar_graph.xgrid.grid_line_color = None
        letter_bar_graph.axis.minor_tick_line_color = None
        letter_bar_graph.outline_line_color = None
        letter_bar_graph.legend.location = "top_left"
        letter_bar_graph.legend.orientation = "horizontal"

        #update div text
        pred.text = "<style>p {font-family:'Lucida Console', Courier, monospace;font-size:24px}</style><p>Prediction: " + prediction + "</p>"
        actual.text = "<style>p {font-family:'Lucida Console', Courier, monospace;font-size:24px}</style><p>Actual: " + targ + "</p>"
        index+=1    
        break
    
  
#runs the program
test()
for epoch in range(1, epoch + 1):
  train(epoch)
  test()





#Generate accuracy graph
graph = figure(plot_width=350, plot_height=275, title="Accuracy Over Time (Lower is Better)")
y = np.arange (1,len(train_losses) + 1)
graph.circle(y = train_losses, x=y, size=7, line_color="#718dbf", fill_color="orange", fill_alpha=0.5)

#Generate Accuracy by letter graph
letter_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
cat = ["correct", "failed"]
colors = ["#718dbf", "#e84d60"]
data = {'letter_list' : letter_list,
        'correct'   : correctByLetter[0],
        'failed'   : correctByLetter[1]}

letter_bar_graph = figure(x_range=letter_list, plot_height=275, plot_width=700, title="Correct By Letter",
           toolbar_location=None, tools="hover", tooltips="$name @letter_list: @$name")


letter_bar_graph.vbar_stack(cat,x='letter_list', width=0.4, source=data, color=colors)

letter_bar_graph.xgrid.grid_line_color = None
letter_bar_graph.y_range.start = 0

#Generate correct bar graph
count = [correct, index-correct]
bar_graph = figure(x_range=cat, plot_height=275, plot_width=350, title="Correct",
           toolbar_location=None, tools="")

bar_graph.vbar(x=cat, top=count, width=0.9, color="#718dbf")

bar_graph.xgrid.grid_line_color = None
bar_graph.y_range.start = 0

#Generate button map
index = 0


button_input = Button(
    label="Next Photo",
    button_type="success",
    width=200
)

button_input.on_click(eval)

#Generate image preview
div_image = Div(text="""<img src="network/static/loading_pic.gif" alt="div_image">""", width=200, height=200)
def update(path):
    div_image.text = "<img src='network/static/{}'>".format(path)
    push_notebook()

#layout the program using Bokeh.layout
doc = curdoc()


div_empty = Div(text="<div>⠀</div>")

left = (column(div_empty,div_image,button_input,pred,actual))
right = (column(row(graph,bar_graph), letter_bar_graph))


doc.add_root(row(left,right))


