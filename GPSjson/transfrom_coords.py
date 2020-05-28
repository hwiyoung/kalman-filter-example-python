from osgeo.osr import SpatialReference, CoordinateTransformation


def geographic2plane(latlon, epsg=32610):
    # Define the Plane Coordinate System (e.g. 5186)
    plane = SpatialReference()
    plane.ImportFromEPSG(epsg)

    # Define the wgs84 system (EPSG 4326)
    geographic = SpatialReference()
    geographic.ImportFromEPSG(4326)

    coord_transformation = CoordinateTransformation(geographic, plane)

    # Check the transformation for a point close to the centre of the projected grid
    xy = coord_transformation.TransformPoint(float(latlon[0]), float(latlon[1]))  # The order: Lat, Lon

    return xy[:2]

def plane2geographic(xy, epsg=32610):
    # Define the Plane Coordinate System (e.g. 5186)
    plane = SpatialReference()
    plane.ImportFromEPSG(epsg)

    # Define the wgs84 system (EPSG 4326)
    geographic = SpatialReference()
    geographic.ImportFromEPSG(4326)

    coord_transformation = CoordinateTransformation(plane, geographic)

    # Check the transformation for a point close to the centre of the projected grid
    latlon = coord_transformation.TransformPoint(float(xy[0]), float(xy[1]))  # The order: x, y

    return latlon[:2]