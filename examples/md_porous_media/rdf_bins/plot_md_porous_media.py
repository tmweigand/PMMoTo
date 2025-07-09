import os
from paraview.simple import (
    CreateView,
    Show,
    GetColorTransferFunction,
    Render,
    SaveScreenshot,
    Hide,
    Delete,
    GetScalarBar,
)
from collections import defaultdict

# === Configuration ===
img_dir = os.path.join(os.path.dirname(__file__), "images")
plot_only_phases = True
img_types = ["img", "uff", "mask"]

# === Gather VTK files ===
vtk_files = [
    os.path.join(img_dir, fname)
    for fname in sorted(os.listdir(img_dir))
    if fname.endswith(".pvti") or fname.endswith(".vti")
]


def get_vtk_reader(vtk_file):
    if vtk_file.endswith(".pvti"):
        from paraview.simple import XMLPImageDataReader

        return XMLPImageDataReader(FileName=[vtk_file])
    elif vtk_file.endswith(".vti"):
        from paraview.simple import XMLImageDataReader

        return XMLImageDataReader(FileName=[vtk_file])
    else:
        raise RuntimeError(f"Unsupported VTK file type: {vtk_file}")


def create_render_view():
    renderView = CreateView("RenderView")
    renderView.Background = [0, 0, 0]
    if hasattr(renderView, "TransparentBackground"):
        renderView.TransparentBackground = 1
    if hasattr(renderView, "UseFXAA"):
        renderView.UseFXAA = 1
    if hasattr(renderView, "OrientationAxesLabelColor"):
        renderView.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
    if hasattr(renderView, "OrientationAxesOutlineColor"):
        renderView.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
    if hasattr(renderView, "OrientationAxesType"):
        renderView.OrientationAxesType = "Arrow"
    return renderView


# === Collect VTK files by img_type ===
img_type_to_files = defaultdict(list)
for vtk_file in vtk_files:
    proc_id = int(vtk_file.split(".")[-2])
    if proc_id == 5:
        continue
    reader = get_vtk_reader(vtk_file)
    reader.UpdatePipeline()
    cell_arrays = list(reader.CellData.keys())
    for img_type in img_types:
        if img_type in cell_arrays:
            img_type_to_files[img_type].append(vtk_file)

# === Run rendering for each img_type ===
for img_type in img_types:
    files = img_type_to_files[img_type]
    if not files:
        continue

    renderView = create_render_view()
    renderView.CameraPosition = [
        1.0864706715473946,
        -0.5719235306801185,
        0.7976751614445559,
    ]
    renderView.CameraFocalPoint = [
        0.19727049767971022,
        0.19727049767971028,
        0.19727049767971058,
    ]
    renderView.CameraViewUp = [
        -0.3961834919646831,
        0.23485491765314753,
        0.8876270660298745,
    ]

    renderView.CameraPosition = [1.086, -0.572, 0.798]
    renderView.CameraFocalPoint = [0.197, 0.197, 0.197]
    renderView.CameraViewUp = [-0.396, 0.235, 0.888]

    displays = []
    lut = None

    for vtk_file in files:
        print(f"Rendering {vtk_file}")
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()
        display = Show(reader, renderView)
        display.ColorArrayName = ["CELLS", img_type]
        lut = GetColorTransferFunction(img_type)
        display.Representation = "Surface"

        if img_type == "img":
            lut.RGBPoints = [0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
            lut.InterpretValuesAsCategories = 1
            lut.Annotations = ["0", "Solid", "1", "Pore"]
            lut.IndexedColors = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
            display.LookupTable = lut
            # elif img_type == "uff":
            #     print("UFF")
            #     lut.RGBPoints = [0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
            #     lut.InterpretValuesAsCategories = 1
            #     lut.Annotations = ["0", "Solid", "1", "Pore"]
            #     lut.IndexedColors = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
            #     display.LookupTable = lut
            # elif img_type == "mask":
            #     print("nask")
            #     lut.RGBPoints = [0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 1.0]
            #     lut.InterpretValuesAsCategories = 1
            #     lut.Annotations = [
            #         "0",
            #         "Solid",
            #         "1",
            #         "Pore Space",
            #         "2",
            #         "Non-Equilibrium Pore Space",
            #     ]
            #     lut.IndexedColors = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
            display.LookupTable = lut

        display.RescaleTransferFunctionToDataRange(False, True)
        displays.append(display)

    # Add a scalar bar (legend) for the current img_type
    if lut is not None:
        scalar_bar = GetScalarBar(lut, renderView)

        scalar_bar.Visibility = 1
        scalar_bar.LookupTable = lut
        scalar_bar.Title = ""
        scalar_bar.ComponentTitle = ""
        scalar_bar.LabelColor = [0, 0, 0]
        scalar_bar.LabelFormat = "%-#6.1f"
        scalar_bar.Orientation = "Horizontal"
        scalar_bar.TitleFontSize = 12
        # scalar_bar.WindowLocation = "Lower Center"
        scalar_bar.WindowLocation = "Upper Right Corner"

    renderView.ResetCamera()
    Render()

    out_png = os.path.join(img_dir, f"{img_type}.png")
    print(f"Saving screenshot to {out_png}")
    SaveScreenshot(
        out_png,
        renderView,
        ImageResolution=[2000, 2000],
        TransparentBackground=True,
    )

    for display in displays:
        Hide(display.Input, renderView)
    Delete(renderView)
    del renderView
