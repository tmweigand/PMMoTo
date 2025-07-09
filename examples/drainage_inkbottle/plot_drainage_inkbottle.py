"""Run this with Paraview python pvpython.

For example, on OSX with version Paraview 5.12.1:
    /Applications/ParaView-5.12.1.app/Contents/bin/pvpython examples/drainage_inkbottle/plot_drainage_inkbottle.py
"""

import os
import paraview.simple as pvs
from collections import defaultdict


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
    renderView = pvs.CreateView("RenderView")
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


def save_screenshot(img_name, render_view, displays):
    """Save a a screenshot"""
    render_view.ResetCamera()
    pvs.Render()
    out_png = os.path.join(os.path.dirname(__file__), f"{img_name}.png")
    print(f"Saving screenshot to {out_png}")
    pvs.SaveScreenshot(
        out_png,
        render_view,
        ImageResolution=[2000, 2000],
        TransparentBackground=True,
    )

    for display in displays:
        pvs.Hide(display.Input, render_view)
    pvs.Delete(render_view)
    del render_view


def pore_space_img(
    files,
):
    """Create an image of the pore space"""
    img_type = "img"
    img_name = "ink_bottle"

    renderView = create_render_view()

    displays = []
    lut = None

    for vtk_file in files:
        print(f"Rendering {vtk_file}")
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()
        pvs.SetActiveSource(reader)  # Make sure reader is active
        pvs.QuerySelect(QueryString=f"({img_type} == 1)", FieldType="CELL", InsideOut=0)
        extracted = pvs.ExtractSelection(Input=reader)
        pvs.Hide(reader, renderView)
        pvs.Hide(extracted, renderView)
        extracted.UpdatePipeline()
        display = pvs.Show(extracted, renderView)
        display.Representation = "Surface"
        renderView.CameraPosition = [-6, 24, 6]
        renderView.CameraFocalPoint = [7, 0, 0]
        renderView.CameraViewUp = [1, 1, -2]

        display.ColorArrayName = ["CELLS", img_type]
        display.SetScalarBarVisibility(renderView, True)

        lut = pvs.GetColorTransferFunction("img")
        lut.RGBPoints = [1, 0, 0, 1]
        lut.IndexedColors = [0, 0, 1]
        lut.Annotations = ["1", "Pore Space and Reservoir"]

        scalar_bar = pvs.GetScalarBar(lut, renderView)
        scalar_bar.Visibility = 1
        scalar_bar.LookupTable = lut
        scalar_bar.Title = ""
        scalar_bar.ComponentTitle = ""
        scalar_bar.LabelColor = [0, 0, 0]
        scalar_bar.TitleFontSize = 14
        scalar_bar.Visibility = 0
        scalar_bar.Orientation = "Horizontal"
        scalar_bar.WindowLocation = "Lower Center"

    save_screenshot(img_name, renderView, displays)


def subdomains_img(
    files,
):
    """Create an image of the pore space"""
    img_type = "img"
    img_name = "subdomains"

    renderView = create_render_view()

    displays = []
    lut = None

    for vtk_file in files:
        print(f"Rendering {vtk_file}")
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()
        pvs.SetActiveSource(reader)  # Make sure reader is active
        display = pvs.Show(reader, renderView)
        display.Representation = "Outline"
        display.AmbientColor = [0.0, 0.0, 1.0]  # Red
        display.DiffuseColor = [0.0, 0.0, 1.0]  # Also red
        display.LineWidth = 4.0
        renderView.CameraPosition = [-6, 24, 6]
        renderView.CameraFocalPoint = [7, 0, 0]
        renderView.CameraViewUp = [1, 1, -2]

    save_screenshot(img_name, renderView, displays)


def cross_section_img(
    files,
):

    img_name = "standard_drainage"

    renderView = create_render_view()
    displays = []
    lut = None

    for vtk_file in files:
        print(f"Rendering {vtk_file}")
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()

        # create a new 'Slice'
        slice1 = pvs.Slice(registrationName="Slice1", Input=reader)

        # Properties modified on slice1.SliceType
        slice1.SliceType.Origin = [0.0, 0.0, 0.0]
        slice1.SliceType.Normal = [0.0, 0.0, 1.0]
        display = pvs.Show(slice1, renderView)
        display.Representation = "Surface"

        display.ColorArrayName = ["CELLS", img_type]
        display.SetScalarBarVisibility(renderView, True)

        # Create and configure color transfer function (LUT)
        lut = pvs.GetColorTransferFunction(img_type)

        lut.RGBPoints = [0, 0.5, 0.5, 0.5, 2, 0, 0, 1, 1, 1, 0, 0]
        lut.Annotations = [
            "0",
            "Solid",
            "1",
            "Non-wetting phase",
            "2",
            "Wetting phase",
        ]
        lut.IndexedColors = [
            0.5,
            0.5,
            0.5,  # gray
            0.0,
            0.0,
            1.0,  # blue
            1.0,
            0.0,
            0.0,  # red
        ]
        lut.InterpretValuesAsCategories = 1

        display.LookupTable = lut
        display.RescaleTransferFunctionToDataRange(False, True)
        displays.append(display)

        scalar_bar = pvs.GetScalarBar(lut, renderView)
        scalar_bar.Visibility = 1
        scalar_bar.LookupTable = lut
        scalar_bar.Title = ""
        scalar_bar.ComponentTitle = ""
        scalar_bar.LabelColor = [0, 0, 0]
        scalar_bar.TitleFontSize = 14
        scalar_bar.WindowLocation = "Lower Right Corner"

    save_screenshot(img_name, renderView, displays)


if __name__ == "__main__":

    # === Configuration ===
    img_dir = os.path.join(os.path.dirname(__file__), "image_proc")
    plot_only_phases = True
    img_types = ["img", "mp_img"]

    # === Gather VTK files ===
    vtk_files = [
        os.path.join(img_dir, fname)
        for fname in sorted(os.listdir(img_dir))
        if fname.endswith(".pvti") or fname.endswith(".vti")
    ]

    # === Collect VTK files by img_type ===
    img_type_to_files = defaultdict(list)
    for vtk_file in vtk_files:
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()
        cell_arrays = list(reader.CellData.keys())
        for img_type in img_types:
            if img_type in cell_arrays:
                img_type_to_files[img_type].append(vtk_file)

    pore_space_img(img_type_to_files["img"])
    subdomains_img(img_type_to_files["img"])
    cross_section_img(img_type_to_files["mp_img"])
