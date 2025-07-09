"""Run this with Paraview python pvpython.

For example, on OSX with version Paraview 5.12.1:
    /Applications/ParaView-5.12.1.app/Contents/bin/pvpython examples/sphere_pack_psd/plot_sphere_pack_psd.py
"""

import os
import copy
import numpy as np
import paraview.simple as pvs
from collections import defaultdict


def get_vtk_reader(vtk_file):
    if vtk_file.endswith(".pvti"):
        print("PArallel")
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
    renderView.UseFXAA = 1
    renderView.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
    renderView.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
    # renderView.OrientationAxesType = "Arrow"

    renderView = initialize_view(renderView)

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


def initialize_view(render_view):
    """Set camera angle and others."""
    render_view.CameraPosition = [
        1.0864706715473946,
        -0.5719235306801185,
        0.7976751614445559,
    ]
    render_view.CameraFocalPoint = [
        0.19727049767971022,
        0.19727049767971028,
        0.19727049767971058,
    ]
    render_view.CameraViewUp = [
        -0.3961834919646831,
        0.23485491765314753,
        0.8876270660298745,
    ]

    render_view.CameraPosition = [1.086, -0.572, 0.798]
    render_view.CameraFocalPoint = [0.197, 0.197, 0.197]
    render_view.CameraViewUp = [-0.396, 0.235, 0.888]

    return render_view


def contour_image(files, img_type, img_name):
    """Create a contour image"""
    renderView = create_render_view()

    displays = []
    lut = None

    for vtk_file in files:
        print(f"Rendering {vtk_file}")
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()
        display = pvs.Show(reader, renderView)
        display.ColorArrayName = ["CELLS", img_type]
        lut = pvs.GetColorTransferFunction(img_type)
        display.Representation = "Surface"

        num_bins = 50
        min_val, max_val = 0.0, img_type_to_range[img_type][1]
        bins = np.linspace(min_val, max_val, num_bins - 1, endpoint=True)
        colors = np.linspace([0, 0, 1], [1, 0, 0], num_bins - 1)  # blue → red

        # Build RGBPoints: [val, R, G, B, val, R, G, B, ...]
        rgb_points = [0, 0.5, 0.5, 0.5]
        for val, rgb in zip(bins, colors):
            rgb_points.extend([val, *rgb])

        lut = pvs.GetColorTransferFunction(img_type)
        lut.RGBPoints = rgb_points
        lut.ColorSpace = "RGB"

        lut.Discretize = 1
        lut.NumberOfTableValues = num_bins
        lut.RescaleTransferFunction(0, max_val)
        display.LookupTable = lut
        scalar_bar = pvs.GetScalarBar(lut, renderView)

        scalar_bar.Visibility = 1
        scalar_bar.LookupTable = lut
        scalar_bar.Title = ""
        scalar_bar.ComponentTitle = ""
        scalar_bar.LabelColor = [0, 0, 0]
        scalar_bar.Orientation = "Horizontal"
        scalar_bar.TitleFontSize = 12
        scalar_bar.WindowLocation = "Upper Right Corner"

        display.SetScalarBarVisibility(renderView, True)
        displays.append(display)

    save_screenshot(img_name, renderView, displays)


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


def pore_space_img(files, img_type, img_name):
    """Create an image of the pore space"""
    renderView = create_render_view()

    displays = []
    lut = None

    for vtk_file in files:
        print(f"Rendering {vtk_file}")
        reader = get_vtk_reader(vtk_file)
        reader.UpdatePipeline()
        display = pvs.Show(reader, renderView)
        display.ColorArrayName = ["CELLS", img_type]
        lut = pvs.GetColorTransferFunction(img_type)
        display.Representation = "Surface"

        lut.RGBPoints = [0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.98, 0.94]
        lut.InterpretValuesAsCategories = 1
        lut.Annotations = ["0", "Solid", "1", "Pore"]
        lut.IndexedColors = [0.5, 0.5, 0.5, 1.0, 0.98, 0.94]
        display.LookupTable = lut

        display.RescaleTransferFunctionToDataRange(False, True)
        displays.append(display)

    # Add a scalar bar (legend) for the current img_type
    if lut is not None:
        scalar_bar = pvs.GetScalarBar(lut, renderView)

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

    save_screenshot(img_name, renderView, displays)


if __name__ == "__main__":

    # === Configuration ===
    img_dir = os.path.join(os.path.dirname(__file__), "image_proc")
    plot_only_phases = True
    img_types = ["img", "dist", "psd", "invert_pm", "invert_dist", "invert_psd"]

    # === Gather VTK files ===
    vtk_files = [
        os.path.join(img_dir, fname)
        for fname in sorted(os.listdir(img_dir))
        if fname.endswith(".pvti") or fname.endswith(".vti")
    ]

    # === Collect VTK files by img_type ===
    img_type_to_files = defaultdict(list)
    img_type_to_files_excluded = defaultdict(list)
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

                print(img_type, vmax, img_type_to_range[img_type][1])

    # Remove proc file for view point
    remove_proc = 5
    img_type_to_files_excluded = copy.deepcopy(img_type_to_files)

    # Remove any file that ends with "_remove_proc.vti"
    for img_type, file_list in img_type_to_files_excluded.items():
        img_type_to_files_excluded[img_type] = [
            f for f in file_list if not f.endswith(f"_{remove_proc}.vti")
        ]

    subdomains_img(img_type_to_files["img"])

    pore_space_img(img_type_to_files_excluded["img"], "img", "pore_space")
    contour_image(img_type_to_files_excluded["dist"], "dist", "distance")
    contour_image(img_type_to_files_excluded["psd"], "psd", "psd")

    pore_space_img(
        img_type_to_files_excluded["invert_pm"], "invert_pm", "inverted_pore_space"
    )
    contour_image(
        img_type_to_files_excluded["invert_dist"], "invert_dist", "invert_distance"
    )
    contour_image(img_type_to_files_excluded["invert_psd"], "invert_psd", "invert_psd")
