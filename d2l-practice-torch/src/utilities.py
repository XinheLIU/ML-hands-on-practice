
from d2l import torch as d2l
import collections
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


class HyperParameters:  #@save
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
    
class ProgressBoard(d2l.HyperParameters):
    """The board that plots data points in animation.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        """
        Adds a new point to the plot and updates the visualization. 
        The points are averaged over `every_n` points before being plotted.
        
        Args:
            x (float): The x-coordinate of the new point.
            y (float): The y-coordinate of the new point.
            label (str): The label associated with the line being drawn.
            every_n (int): The interval at which the points are averaged and plotted.
        """
        # Define a named tuple `Point` to represent individual points
        Point = collections.namedtuple('Point', ['x', 'y'])
        
        # Initialize storage for raw points and averaged data if not already set
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()  # Store raw data points for averaging
            self.data = collections.OrderedDict()        # Store averaged data for plotting
        
        # Initialize data structures for the specified label if not already present
        if label not in self.raw_points:
            self.raw_points[label] = []  # List to hold raw points for this label
            self.data[label] = []        # List to hold averaged points for this label
        
        # Add the new point to the list of raw points
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        
        # Only process the points if their count matches `every_n`
        if len(points) != every_n:
            return  # Wait until we have enough points to average
        
        # Compute the mean of the x and y coordinates of the points
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]), 
                        mean([p.y for p in points])))
        
        # Clear the raw points after averaging
        points.clear()
        
        # Skip plotting if display is disabled
        if not self.display:
            return
        
        # Use SVG format for better display resolution in plots
        d2l.use_svg_display()
        
        # Initialize the figure if it doesn't exist
        if self.fig is None:
            self.fig = d2l.plt.figure(figsize=self.figsize)
        
        # Prepare the plot lines and labels for the updated data
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(d2l.plt.plot(
                [p.x for p in v],  # Extract x-coordinates
                [p.y for p in v],  # Extract y-coordinates
                linestyle=ls,      # Line style for this label
                color=color         # Line color for this label
            )[0])
            labels.append(k)  # Store the label
        
        # Set axes properties, such as limits, labels, and scales
        axes = self.axes if self.axes else d2l.plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        
        # Add a legend to the plot
        axes.legend(plt_lines, labels)
        
        # Display the updated figure and clear the output for real-time updates
        display.display(self.fig)
        display.clear_output(wait=True)