from ipywidgets import Output, GridspecLayout
from IPython import display

filepaths = ['../datatest/videos/SCD_circulos_1sentido_contrario.avi', '../datatest/videos/SCE_bur_11_girando.avi', '../datatest/videos/SCE_circulos_girando_3.avi']
grid = GridspecLayout(1, len(filepaths))

for i, filepath in enumerate(filepaths):
    out = Output()
    with out:
        display.display(display.Video(filepath, embed=True))
    grid[0, i] = out

grid