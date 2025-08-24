# requires vtk and numpy
import vtk
import numpy as np

class CameraModel:
    """
    Generates a VTK model of a camera
    It's a box with a cone looking along the Z axis
    """
    def __init__(self, size, colour):
        self.size = size
        self.box = vtk.vtkCubeSource()
        self.box.SetXLength(self.size)
        self.box.SetYLength(self.size)
        self.box.SetZLength(self.size)
        self.box.SetCenter(0.0, 0.0, -size/2.0)
        self.box.Update()
        self.cone = vtk.vtkConeSource()
        self.cone.SetCenter(0.0, 0.0, size/2.0)
        self.cone.SetDirection(0.0, 0.0, -1.0)
        self.cone.SetHeight(size)
        self.cone.SetRadius(size/2.0)
        self.cone.Update()
        self.polydata = vtk.vtkAppendPolyData()
        # Not sure if this copy is needed - copied from example
        # https://vtk.org/Wiki/VTK/Examples/Python/Filtering/CombinePolyData
        input1 = vtk.vtkPolyData()
        input1.ShallowCopy(self.cone.GetOutput())
        input2 = vtk.vtkPolyData()
        input2.ShallowCopy(self.box.GetOutput())
        if vtk.VTK_MAJOR_VERSION <= 5:
            self.polydata.AddInputConnection(input1.GetProducerPort())
            self.polydata.AddInputConnection(input2.GetProducerPort())
        else:
            self.polydata.AddInputData(input1)
            self.polydata.AddInputData(input2)
        self.polydata.Update()
        ncellsBox = self.box.GetOutput().GetNumberOfCells()
        print("N Box cells: {}".format(ncellsBox))
        ncellsCone = self.cone.GetOutput().GetNumberOfCells()
        print("N Cone cells: {}".format(ncellsCone))
        n_cells = self.polydata.GetOutput().GetNumberOfCells()
        print("Total cells: {}".format(n_cells))
        colours = vtk.vtkUnsignedCharArray()
        colours.SetName("colors")
        colours.SetNumberOfComponents(3)
        colours.SetNumberOfTuples(n_cells)
        #colours.InsertNextValue(colour)
        #colours.InsertNextTuple(colour)
        for i in range(n_cells):
            colours.InsertTypedTuple(i, colour)
        self.polydata.GetOutput().GetCellData().SetScalars(colours)

        self.transform = vtk.vtkTransform()
        self.transformFilter = vtk.vtkTransformPolyDataFilter()
        self.transformFilter.SetTransform(self.transform)
        self.transformFilter.SetInputData(self.polydata.GetOutput())

    # Set the transformation from OpenCV matrices
    def set_transform_matrix(self, R, T):
        mat = vtk.vtkMatrix4x4()
        for j in range(3):
            for i in range(3):
                mat.SetElement(i, j, R[i][j])
            mat.SetElement(j, 3, T[j])
        self.transform.SetMatrix(mat)
        # Not sure, but inversion seems to put the left camera on the left
        self.transform.Inverse()
        self.transform.Update()
        self.transformFilter.Update()

    def get_model(self):
        self.transformFilter.Update()
        return self.transformFilter.GetOutput()


class StereoCameraModel:
    """
     Generates a VTK model of a two stereo cameras
     """
    def __init__(self, R, T):
        # Rough size - a bit less than the eye separation
        dist = np.sqrt(np.sum(np.square(T)))
        size = dist / 4.0
        self.left_camera = CameraModel(size, (255, 0, 0))
        self.right_camera = CameraModel(size, (0, 255, 0))
        self.right_camera.set_transform_matrix(R, T)
        print("Transform: {}".format(self.right_camera.transform))
        # Append the two camera models
        self.polydata = vtk.vtkAppendPolyData()
        # Not sure if this copy is needed - copied from example
        # https://vtk.org/Wiki/VTK/Examples/Python/Filtering/CombinePolyData
        self.polydata.AddInputData(self.left_camera.get_model())
        self.polydata.AddInputData(self.right_camera.get_model())
        self.polydata.Update()

    def get_model(self):
        return self.polydata.GetOutput()
