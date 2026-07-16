
import dash_molstar

def create_line(coord1, coord2, label, color='blue', radius=.6, **kwargs):
    return dash_molstar.utils.shapes.create_cylinder(
        start=tuple(float(x) for x in coord1),
        end=tuple(float(x) for x in coord2),
        label=label,
        color='blue',
        radius=radius,
        **kwargs,
    )
