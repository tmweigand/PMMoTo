"""Run this with Paraview python pvpython.

For example, on OSX with version Paraview 5.12.1:
    /Applications/ParaView-5.12.1.app/Contents/bin/pvpython examples/md_porous_media/plot_md_porous_media.py
"""

import os
import copy
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


def plot_pore_space(files, img_type, img_name, name, color):
    """Plot the pore space"""
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
        if len(color) > 3:
            _color = color[0:3] + [2.0] + color[4:]
        else:
            _color = color
        lut.RGBPoints = [0.0, 0.7137, 0.7137, 0.7137, 1.0] + _color
        lut.InterpretValuesAsCategories = 1
        lut.Annotations = ["0", "Solid"] + [
            item for i, label in enumerate(name, start=1) for item in (str(i), label)
        ]
        # lut.Annotations = ["0", "Solid", "1"] + name
        lut.IndexedColors = [0.7137, 0.7137, 0.7137] + color
        display.LookupTable = lut
        display.RescaleTransferFunctionToDataRange(False, True)
        displays.append(display)

    scalar_bar = pvs.GetScalarBar(lut, renderView)
    scalar_bar.Visibility = 1
    scalar_bar.Title = ""
    scalar_bar.ComponentTitle = ""
    scalar_bar.LabelColor = [0, 0, 0]
    scalar_bar.Orientation = "Horizontal"
    scalar_bar.WindowLocation = "Upper Center"
    # scalar_bar.WindowLocation = "Any Location"
    # scalar_bar.Position = [0.05, 0.05]
    scalar_bar.ScalarBarLength = 0.75
    scalar_bar.LabelFontSize = 9
    scalar_bar.LookupTable = lut

    save_screenshot(img_name, renderView, displays)


# # === Run rendering for each img_type ===
# for img_type in img_types:
#     files = img_type_to_files[img_type]
#     if not files:
#         continue

#     renderView = create_render_view()
#     renderView.CameraPosition = [
#         1.0864706715473946,
#         -0.5719235306801185,
#         0.7976751614445559,
#     ]
#     renderView.CameraFocalPoint = [
#         0.19727049767971022,
#         0.19727049767971028,
#         0.19727049767971058,
#     ]
#     renderView.CameraViewUp = [
#         -0.3961834919646831,
#         0.23485491765314753,
#         0.8876270660298745,
#     ]

#     renderView.CameraPosition = [1.086, -0.572, 0.798]
#     renderView.CameraFocalPoint = [0.197, 0.197, 0.197]
#     renderView.CameraViewUp = [-0.396, 0.235, 0.888]

# displays = []
# lut = None

# for vtk_file in files:
#     print(f"Rendering {vtk_file}")
#     reader = get_vtk_reader(vtk_file)
#     reader.UpdatePipeline()
#     display = pvs.Show(reader, renderView)
#     display.ColorArrayName = ["CELLS", img_type]
#     lut = pvs.GetColorTransferFunction(img_type)
#     display.Representation = "Surface"

#     if img_type == "img":
#         lut.RGBPoints = [
#             0.0,
#             0.7137,
#             0.7137,
#             0.7137,
#             1.0,
#             0.0,
#             0.0,
#             0.482,
#         ]
#         lut.InterpretValuesAsCategories = 1
#         lut.Annotations = ["0", "Solid", "1", "PMF pore space"]
#         lut.IndexedColors = [
#             0.7137,
#             0.7137,
#             0.7137,
#             0.0,
#             0.0,
#             0.482,
#         ]
#         display.LookupTable = lut
#     elif img_type == "uff":
#         lut.RGBPoints = [
#             0.0,
#             0.7137,
#             0.7137,
#             0.7137,
#             1.0,
#             1.0,
#             0.929,
#             0.878,
#         ]
#         lut.InterpretValuesAsCategories = 1
#         lut.Annotations = ["0", "Solid", "1", "UFF pore space"]
#         lut.IndexedColors = [
#             0.7137,
#             0.7137,
#             0.7137,
#             1.0,
#             0.929,
#             0.878,
#         ]
#         display.LookupTable = lut
#     elif img_type == "mask":
#         lut.RGBPoints = [
#             0.0,
#             0.7137,
#             0.7137,
#             0.7137,
#             1.0,
#             1.0,
#             0.929,
#             0.878,
#             2.0,
#             0.0,
#             0.0,
#             0.482,
#         ]
#         lut.InterpretValuesAsCategories = 1
#         lut.Annotations = [
#             "0",
#             "Solid",
#             "1",
#             "Equilibrium pore space",
#             "2",
#             "Non-equilibrium pore space",
#         ]
#         lut.IndexedColors = [
#             0.7137,
#             0.7137,
#             0.7137,
#             1.0,
#             0.929,
#             0.878,
#             0.0,
#             0.0,
#             0.482,
#         ]
#         display.LookupTable = lut

#     display.RescaleTransferFunctionToDataRange(False, True)
#     displays.append(display)

# # Add a scalar bar (legend) for the current img_type
# if lut is not None:
#     scalar_bar = pvs.GetScalarBar(lut, renderView)

#     scalar_bar.Visibility = 1
#     scalar_bar.Title = ""
#     scalar_bar.ComponentTitle = ""
#     scalar_bar.LabelColor = [0, 0, 0]
#     if img_type == "mask":
#         scalar_bar.Orientation = "Horizontal"
#         scalar_bar.WindowLocation = "Upper Center"
#         # scalar_bar.WindowLocation = "Any Location"
#         # scalar_bar.Position = [0.05, 0.05]
#         scalar_bar.ScalarBarLength = 0.75
#     else:
#         scalar_bar.Orientation = "Horizontal"
#         scalar_bar.WindowLocation = "Upper Center"
#         scalar_bar.ScalarBarLength = 0.45
#     scalar_bar.LabelFontSize = 9
#     scalar_bar.LookupTable = lut

if __name__ == "__main__":

    # === Configuration ===
    img_dir = os.path.join(os.path.dirname(__file__), "image_proc")
    img_types = ["img", "uff", "mask"]

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

    pmf_color = [0.0, 0.0, 0.482]
    pmf_name = ["PMF pore space"]
    plot_pore_space(
        img_type_to_files_excluded["img"], "img", "pmf_pore_space", pmf_name, pmf_color
    )

    uff_color = [1.0, 0.929, 0.878]
    uff_name = ["UFF pore space"]
    plot_pore_space(
        img_type_to_files_excluded["uff"], "uff", "uff_pore_space", uff_name, uff_color
    )

    mask_color = uff_color + pmf_color
    mask_name = ["Non-equilibrium pore space", "Equilibrium pore space"]
    plot_pore_space(
        img_type_to_files_excluded["mask"],
        "mask",
        "comparison_pore_space",
        mask_name,
        mask_color,
    )
