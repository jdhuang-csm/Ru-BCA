# -*- coding: utf-8 -*-


import sys
import os
from bokeh.application.handlers.function import FunctionHandler
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, Label
from bokeh.palettes import d3
import pandas as pd
import numpy as np


palette = d3['Category20'][16]


df = pd.read_csv(sys.argv[1], skiprows=19, skipinitialspace=True)

df['hours'] = df['Elapsed time']/3600
df['Total Flow'] = df['N2-flow'] + df['H2-flow']
df = df.drop(df.columns[[2, 6, 11]], axis=1)

plotsource = ColumnDataSource(df)



pheight = 230
pwidth = 650
ptools="box_select,box_zoom,pan,wheel_zoom,hover,save,reset"

column_names = ['Elapsed time','Pressure','LJtemp(C)','Inner-Temp',
                'Outer-Temp','N2-set','N2-flow','H2-set','H2-flow',
                'pH','NH3_ppm','NH3_Prod_rate','In222Temp']
    

#------------------------Temperature Plot-------------------------

Thoverinfo = [("Sensor", "$name"),
         ("Temp", "$y{0.0} C"),
         ('Time', '$x{0.00} hrs')]

Tfig = figure(title='Temperature',
         output_backend='webgl',
         x_axis_label='Time (hrs)',
         y_axis_label='Temperature (C)',
         plot_width=pwidth,
         plot_height=pheight,
         tools=ptools,
         tooltips=Thoverinfo,
         active_drag="box_zoom",
         )

Tfig.line(x='hours',
          y=column_names[4],
          source=plotsource,
          line_width=1,
          color=palette[2],
          name=column_names[4]
          )


#-------------------------Pressure Plot-----------------

Phoverinfo = [
     ("Pressure", "$y{0.0} psi"),
     ('Time', '$x{0.00} hrs')]

Pfig = figure(title='Pressure',
             output_backend='webgl',
             x_axis_label='Time (hrs)',
             y_axis_label='Pressure (psi)',
             plot_width=pwidth,
             plot_height=pheight,
             x_range=Tfig.x_range,
             tools=ptools,
             active_drag="box_zoom",
             tooltips=Phoverinfo
             )

Pfig.line(x='hours',
             y=column_names[1],
             source=plotsource,
             line_width=1,
             line_color=palette[8],
             name=column_names[1]
              )


#------------------------------Flow Plot------------------

Fhoverinfo = [("Sensor", "$name"),
         ('Flow', "$y{0} sLm"),
         ('Time', '$x{0.00} hrs')]


Ffig = figure(title='Flow',
             output_backend='webgl',
             x_axis_label='Time (hrs)',
             y_axis_label='Flow (sLm)',
             plot_width=pwidth,
             plot_height=pheight,
             x_range=Tfig.x_range,
             tools=ptools,
             active_drag="box_zoom",
             tooltips=Fhoverinfo
             )

for i in range(4):
    Ffig.line(x='hours',
         y=column_names[i+5],
         source=plotsource,
         line_width=1,
         line_color=palette[i+4],
         name=column_names[i+5])
         
Ffig.line(x='hours',
     y='Total Flow',
     source=plotsource,
     line_width=1,
     line_color=palette[10],
     name='Total Flow')



#-------------------------Rate Plot------------------

Rhoverinfo = [
         ('Rate', "$y{0.00} mmol/(g hr)"),
         ('Time', '$x{0.00} hrs')]

Rfig = figure(title='NH3 Production Rate',
             output_backend='webgl',
         x_axis_label='Time (hrs)',
         y_axis_label='Rate (mmol/g-hr)',
         y_range=(0,225),
         plot_width=pwidth,
         plot_height=pheight,
         x_range=Tfig.x_range,
         tools=ptools,
         active_drag="box_zoom",
         tooltips=Rhoverinfo
         )


Rfig.line(x='hours',
         y='NH3_Prod_rate',
         source=plotsource,
         line_width=1,
         line_color=palette[14],
         name='NH3 Production Rate'
         )
             

#-------------------------Conversion Plot------------------

Choverinfo = [
         ('PPM', "$y{0.0}"),
         ('Time', '$x{0.00} hrs')]

Cfig = figure(title='NH3 Conversion',
             output_backend='webgl',
         x_axis_label='Time (hrs)',
         y_axis_label='Conversion (ppm)',
         plot_width=pwidth,
         plot_height=pheight,
         x_range=Tfig.x_range,
         y_range=(0,7000),
         tools=ptools,
         active_drag="box_zoom",
         tooltips=Choverinfo
         )


Cfig.line(x='hours',
         y='NH3_ppm',
         source=plotsource,
         line_width=1,
         line_color=palette[0],
         name='NH3 ppm'
         )
         
#-------------------------Conv/Temp Plot------------------

CThoverinfo = [
         ('PPM', "$y{0.0}"),
         ('Temp', '$x{0.00} C')]

CTfig = figure(title='NH3 Conversion vs. Temp',
         output_backend='webgl',
         x_axis_label='Temperature (C)',
         x_range=(0,650),
         y_axis_label='Conversion (ppm)',
         y_range=(0,7000),
         plot_width=pwidth,
         plot_height=pheight,
         tools=ptools,
         active_drag="box_zoom",
         tooltips=CThoverinfo
         )


CTfig.circle(x='Outer-Temp',
         y='NH3_ppm',
         source=plotsource,
         size=2,
         color=palette[0],
         alpha=0.1,
         name='NH3 ppm'
         )


#-------------------------Rate/Flow Plot------------------

RFhoverinfo = [
         ('Rate', "$y{0.0} mmol/g-hr"),
         ('Flow', '$x{0.00} sLm')]

RFfig = figure(title='NH3 Rate vs. Flow',
         output_backend='webgl',
         x_axis_label='Flow (sLm)',
         y_range=(0,225),
         y_axis_label='Rate (mmol/g-hr)',
         plot_width=pwidth,
         plot_height=pheight,
         tools=ptools,
         active_drag="box_zoom",
         tooltips=RFhoverinfo
         )


RFfig.circle(x='Total Flow',
         y='NH3_Prod_rate',
         source=plotsource,
         size=2,
         color=palette[14],
         alpha=0.1,
         name='Rate'
         )

#-------------------------Notes Panel----------------------

nfig = figure(title='Notes',
         output_backend='webgl',
         plot_width=pwidth,
         plot_height=pheight,
         tools=ptools,
         active_drag="box_zoom",
         )

fp = open(sys.argv[1])
for i, line in enumerate(fp):
    if i == 9:
        notes1 = line
    elif i == 10:
        notes2 = line
    elif i > 10:
        break
fp.close()

label1 = Label(x=0, y=0.8, text=notes1, text_font_size='8pt')
label2 = Label(x=0, y=0.6, text=notes2, text_font_size='8pt')

nfig.add_layout(label1)
nfig.add_layout(label2)

nfig.circle(x=[0,1],
         y=[0,1],
         size=2,
         color=palette[1],
         alpha=0.0,
         name='Empty'
         )


grid = gridplot([[Tfig, Rfig], [Pfig, RFfig], [Ffig, Cfig], [nfig, CTfig]])
    
output_file(sys.argv[1] + ".html", title=sys.argv[1])

show(grid)





