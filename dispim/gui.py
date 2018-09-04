import wx


class DSTGui(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: DSTGui.__init__
        kwds["style"] = kwds.get("style", 0) | wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.SetSize((491, 309))

        # Menu Bar
        self.frame_menubar = wx.MenuBar()
        wxglade_tmp_menu = wx.Menu()
        self.frame_menubar.Append(wxglade_tmp_menu, "File")
        self.SetMenuBar(self.frame_menubar)
        # Menu Bar end
        self.list_box_1 = wx.ListBox(self, wx.ID_ANY, choices=["c"])
        self.button_2 = wx.Button(self, wx.ID_ANY, "+")
        self.button_3 = wx.Button(self, wx.ID_ANY, "V")

        self.__set_properties()
        self.__do_layout()
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: DSTGui.__set_properties
        self.SetTitle("frame")
        self.button_2.SetMinSize((34, 34))
        self.button_3.SetMinSize((34, 34))
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: DSTGui.__do_layout
        sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_3 = wx.BoxSizer(wx.VERTICAL)
        sizer_4 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_3.Add(self.list_box_1, 0, wx.EXPAND | wx.SHAPED | wx.TOP, 0)
        sizer_4.Add(self.button_2, 0, 0, 0)
        sizer_4.Add(self.button_3, 0, 0, 0)
        sizer_3.Add(sizer_4, 9, wx.EXPAND, 0)
        sizer_2.Add(sizer_3, 1, wx.EXPAND, 0)
        sizer_2.Add((0, 0), 0, 0, 0)
        self.SetSizer(sizer_2)
        self.Layout()
        # end wxGlade

    def add_process(self, event):  # wxGlade: DSTGui.<event_handler>
        self.list_box_1.InsertItems(['test'], self.list_box_1.GetSelection()+1)
        # event.Skip()

# end of class DSTGui

class DSTApp(wx.App):
    def OnInit(self):
        self.frame = DSTGui(None, wx.ID_ANY, "dST GUI")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True
