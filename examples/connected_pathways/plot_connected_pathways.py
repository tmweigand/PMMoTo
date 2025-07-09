"""Run this with Paraview python pvpython.

For example, on OSX with version Paraview 5.12.1:
    /Applications/ParaView-5.12.1.app/Contents/bin/pvpython examples/connected_pathways/plot_connected_pathways.py
"""

import os
import paraview.simple as pvs
import random
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


def add_inlet_out(render_view):
    """Add inlet and outlet labels"""
    inlet_title = pvs.Text()
    inlet_title.Text = "Inlet"
    titleDisplay = pvs.Show(inlet_title, render_view)
    titleDisplay.WindowLocation = "Any Location"
    titleDisplay.Position = [0.12, 0.4615]
    titleDisplay.FontSize = 16
    titleDisplay.Color = [0, 0, 0]
    titleDisplay.Orientation = 90.0

    outlet_title = pvs.Text()
    outlet_title.Text = "Outlet"
    titleDisplay = pvs.Show(outlet_title, render_view)
    titleDisplay.WindowLocation = "Any Location"
    titleDisplay.Position = [0.85, 0.43]
    titleDisplay.FontSize = 16
    titleDisplay.Color = [0, 0, 0]
    titleDisplay.Orientation = 90.0


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


def id_to_color(proc_id, max_id=8):
    # Map proc_id to a distinct RGB color using HSV cycling
    import colorsys

    hue = (proc_id % max_id) / max_id
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return [r, g, b]


def subdomains_img(
    files,
):
    """Create an image of the pore space"""
    img_type = "img"
    img_name = "subdomains"

    renderView = create_render_view()
    add_inlet_out(renderView)

    displays = []
    lut = None

    for vtk_file in files:
        print(f"Rendering {vtk_file}")
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()
        pvs.SetActiveSource(reader)  # Make sure reader is active
        display = pvs.Show(reader, renderView)
        display.Representation = "Outline"
        proc_id = int(vtk_file.split("_")[-1][0])
        color = id_to_color(proc_id)
        display.AmbientColor = color
        display.DiffuseColor = color
        # display.AmbientColor = [0.0, 0.0, 1.0]
        # display.DiffuseColor = [0.0, 0.0, 1.0]
        display.LineWidth = 4.0

    save_screenshot(img_name, renderView, displays)


def plot_pore_space(files, img_type, img_name):
    """Plot the pore space"""
    renderView = create_render_view()
    add_inlet_out(renderView)
    displays = []
    lut = None  # Store LUT for legend
    for vtk_file in files:
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()
        display = pvs.Show(reader, renderView)
        display.ColorArrayName = ["CELLS", img_type]
        display.Representation = "Surface"
        lut = pvs.GetColorTransferFunction(img_type)
        lut.RGBPoints = [0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.98, 0.96]
        lut.InterpretValuesAsCategories = 1
        lut.Annotations = ["0", "Solid", "1", "Pore"]
        lut.IndexedColors = [0.5, 0.5, 0.5, 1.0, 0.98, 0.96]
        display.LookupTable = lut

        displays.append(display)

    scalar_bar = pvs.GetScalarBar(lut, renderView)
    scalar_bar.Visibility = 1
    scalar_bar.LookupTable = lut
    scalar_bar.Title = ""  # img_type.replace("_", " ").title()
    scalar_bar.ComponentTitle = ""
    # scalar_bar.TitleColor = [0, 0, 0]
    scalar_bar.LabelColor = [0, 0, 0]
    scalar_bar.Orientation = "Horizontal"
    scalar_bar.TitleFontSize = 14
    scalar_bar.WindowLocation = "Lower Center"

    save_screenshot(img_name, renderView, displays)


def plot_pore_space_connections(files, img_files, img_type, img_name, title, color):
    """Plot the pore space"""
    renderView = create_render_view()
    add_inlet_out(renderView)
    displays = []
    lut = None  # Store LUT for legend

    for vtk_file in img_files:
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()
        display = pvs.Show(reader, renderView)
        display.ColorArrayName = ["CELLS", "img"]
        display.Representation = "Surface"
        lut = pvs.GetColorTransferFunction("img")
        lut.RGBPoints = [0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.98, 0.96]
        lut.InterpretValuesAsCategories = 1
        lut.Annotations = ["0", "Solid", "1", "Pore"] + ["2", title]
        lut.IndexedColors = [0.5, 0.5, 0.5, 1.0, 0.98, 0.96] + color
        display.LookupTable = lut
        displays.append(display)

    for vtk_file in files:
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()

        pvs.QuerySelect(QueryString=f"({img_type} == 1)", FieldType="CELL", InsideOut=0)
        extracted = pvs.ExtractSelection(Input=reader)

        pvs.Hide(reader, renderView)
        display = pvs.Show(extracted, renderView)
        display.DiffuseColor = color
        extracted.UpdatePipeline()

        display.Representation = "Surface"
        lut = pvs.GetColorTransferFunction(img_type)

        lut.InterpretValuesAsCategories = 1
        lut.Annotations = ["0", "Solid", "1", "Pore"] + ["2", title]
        lut.IndexedColors = [0.5, 0.5, 0.5, 1.0, 0.98, 0.96] + color
        display.LookupTable = lut
        displays.append(display)

    scalar_bar = pvs.GetScalarBar(lut, renderView)
    scalar_bar.Visibility = 1
    scalar_bar.LookupTable = lut
    scalar_bar.Title = ""
    scalar_bar.ComponentTitle = ""
    # scalar_bar.TitleColor = [0, 0, 0]
    scalar_bar.LabelColor = [0, 0, 0]
    scalar_bar.Orientation = "Horizontal"
    scalar_bar.TitleFontSize = 14
    scalar_bar.WindowLocation = "Lower Center"

    save_screenshot(img_name, renderView, displays)


def plot_labels(files, img_type, img_name):
    """Plot the pore space labels"""
    renderView = create_render_view()
    add_inlet_out(renderView)
    displays = []
    lut = None  # Store LUT for legend
    for vtk_file in files:
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()
        display = pvs.Show(reader, renderView)
        display.ColorArrayName = ["CELLS", img_type]
        display.Representation = "Surface"
        lut = pvs.GetColorTransferFunction(img_type)
        # Set solid
        rgb_points = [0.0, 0.5, 0.5, 0.5]

        lut.Annotations = ["0", "Solid"]
        lut.IndexedColors = [0.5, 0.5, 0.5]

        for n in range(1, int(img_type_to_range[img_type][1])):
            random.seed(n)
            random_color = [
                n,
                random.random(),
                random.random(),
                random.random(),
            ]
            rgb_points += random_color

            lut.RGBPoints = rgb_points
            display.LookupTable = lut
            display.RescaleTransferFunctionToDataRange(False, True)

        displays.append(display)

    save_screenshot(img_name, renderView, displays)


if __name__ == "__main__":

    # === Configuration ===
    img_dir = os.path.join(os.path.dirname(__file__), "image_proc")

    vtk_files = [
        os.path.join(img_dir, fname)
        for fname in sorted(os.listdir(img_dir))
        if fname.endswith(".pvti") or fname.endswith(".vti")
    ]

    img_types = [
        "img",  # binary
        "cc",  # labeled, NOT binary
        "inlet_img",  # binary
        "outlet_img",  # binary
        "inlet_outlet_img",  # binary
        "isolated_internal",  # binary
    ]

    img_type_to_files = defaultdict(list)
    img_type_to_range = defaultdict(lambda: [int(0), int(-100000)])  # [min, max]
    for vtk_file in vtk_files:
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()
        cell_arrays = list(reader.CellData.keys())
        for img_type in img_types:
            if img_type in cell_arrays:
                img_type_to_files[img_type].append(vtk_file)

                # Get the min and max for this array
                data_array = reader.CellData[img_type]
                vmin, vmax = data_array.GetRange()

                # Update global min/max
                img_type_to_range[img_type][0] = min(
                    img_type_to_range[img_type][0], vmin
                )
                img_type_to_range[img_type][1] = max(
                    img_type_to_range[img_type][1], vmax
                )

    subdomains_img(img_type_to_files["img"])
    plot_pore_space(img_type_to_files["img"], "img", "pore_space")
    plot_pore_space_connections(
        img_type_to_files["inlet_img"],
        img_type_to_files["img"],
        "inlet_img",
        "inlet_pore_space",
        "Inlet",
        [0, 0, 1],
    )
    plot_pore_space_connections(
        img_type_to_files["outlet_img"],
        img_type_to_files["img"],
        "outlet_img",
        "outlet_pore_space",
        "Outlet",
        [0.251, 0.878, 0.816],
    )
    plot_pore_space_connections(
        img_type_to_files["inlet_outlet_img"],
        img_type_to_files["img"],
        "inlet_outlet_img",
        "inlet_outlet_pore_space",
        "Connected",
        [0.8, 0.4, 0.0],
    )
    plot_pore_space_connections(
        img_type_to_files["isolated_internal"],
        img_type_to_files["img"],
        "isolated_internal",
        "isolated_pore_space",
        "Isolated",
        [1.0, 0, 0],
    )
    plot_labels(
        img_type_to_files["cc"],
        "cc",
        "pore_space_labels",
    )
