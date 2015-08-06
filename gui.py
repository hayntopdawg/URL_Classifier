import Tkinter
import os
import dataset
from classifier import load_voter
from processor import process_url

class URL_Class_tk(Tkinter.Tk):
    def __init__(self, parent):
        # Derives from Tkinter.Tk, so we have to call
        # the Tkinter.Tk constructor (Tkinter.Tk.__init__())
        Tkinter.Tk.__init__(self, parent)
        
        # When building any type of GUI component, it's
        # good habit to keep a reference to our parent
        self.parent = parent
        
        # Usually best to have the portion of code which
        # creates all the GUI elements (button, text fields...)
        # separate from the logic of the program.
        self.initialize()
    
    # We will create all our widgets (buttons, text field, etc.)
    # in this method
    def initialize(self):
        # Initialize grid layout manager
        self.grid()

        self.entryVariable = Tkinter.StringVar()
        # Create the widget
        self.entry = Tkinter.Entry(self, textvariable=self.entryVariable)
        # When a cell grows larger than the widget is contains,
        # you can ask the widget to stick to some edges of the cell.
        # That's the sticky='EW'.
        # (E=east (left), W=West (right), N=North (top), S=South (bottom))
        # We specified 'EW', which means the widget will try to stick to
        # both left and right edges of its cell.
        self.entry.grid(column=0, row=0, sticky='EW')
        # <Return> event handler
        self.entry.bind("<Return>", self.OnPressEnter)
        self.entryVariable.set(u"Enter URL here.")

        # Load classifier (Choose a path where the voter is saved)
        path = os.path.join(os.path.expanduser('~'), 'OneDrive\\RPI\\Summer Project\\URL Classifier\\Dataset\\Trial_03')
        self.voter = load_voter(path)
        test = dataset.from_npy(path, 'test.npy')
        test_X, test_y = test[:, :-1], test[:, -1]
        self.voter.confusion_matrix(test_X, test_y)

        ### Button ###
        # Note that in this case, we do not keep a reference to the button
        # (because we will not read or alter its value later)
        button = Tkinter.Button(self, text=u"Submit",
                                command=self.OnButtonClick)
        button.grid(column=1, row=0)

        ### Label ###
        self.labelVariable = Tkinter.StringVar()
        # White text on a blue background
        self.label = Tkinter.Label(self, textvariable=self.labelVariable,
                                   anchor="w", fg="white", bg="blue")
        self.label.grid(column=0, row=1, columnspan=2, sticky='EW')
        self.labelVariable.set(u"Welcome to URL Classifier!")

        # Enable resizing
        self.grid_columnconfigure(0,weight=1)
        # Adding constraint - Prevent vertical resizing
        self.resizable(width=True, height=False)
        # Set window to not grow and shrink automatically
        self.update()
        self.geometry(self.geometry())
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

    def OnButtonClick(self):
        # self.labelVariable.set( self.entryVariable.get()+" (You clicked the button)" )
        url = self.entryVariable.get()
        try:
            processed_url = process_url(url).values()
            (pred, conf) = self.voter.vote(processed_url)
            url_class = "Malicious" if pred == 1 else "Benign"
            self.labelVariable.set("URL Classified as {}, Confidence: {:.4f}%".format(url_class, conf * 100))
        except:
            self.labelVariable.set("Not a valid URL")
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

    def OnPressEnter(self,event):
        # self.labelVariable.set( self.entryVariable.get()+" (You pressed ENTER)" )
        url = self.entryVariable.get()
        try:
            processed_url = process_url(url).values()
            (pred, conf) = self.voter.vote(processed_url)
            url_class = "Malicious" if pred == 1 else "Benign"
            self.labelVariable.set("URL Classified as {}, Confidence: {:.4f}%".format(url_class, conf * 100))
        except:
            self.labelVariable.set("Not a valid URL")
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

# Create a main which is executed when the program is run 
# from the command-line
if __name__ == "__main__":
    # In Tkinter, we instantiate our class (app=simpleapp_tk()).
    # We give it no parent (None), because it's the first GUI
    # element we build.
    app = URL_Class_tk(None)
    app.title("URL Classifier")
    
    # Event-driven programming 
    # (Because the program will do nothing but wait for events,
    # and only react when it receives an event.)
    app.mainloop()
