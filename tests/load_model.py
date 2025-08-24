from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
import vtk
from sksurgeryvtk.widgets.vtk_overlay_window import VTKOverlayWindow
from vtkmodules.vtkIOLegacy import vtkPolyDataReader

def load_vtk_model(filename):
    reader = vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def main():
    #print("Before QApplication")
    app = QApplication(sys.argv)
    #print("After QApplication")

    base_dir = Path(__file__).resolve().parent
    kidney_model_path = base_dir / 'scikit-surgeryvtk' / 'tests' / 'data' / 'models' / 'Kidney' / 'kidney.vtk'
    # kidney_model_path = base_dir / 'serv-ct' / 'CT' / '001' / 'Anatomy.vtk'
    vtk_model = load_vtk_model(str(kidney_model_path))

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vtk_model)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    window = VTKOverlayWindow()
    window.add_vtk_actor(actor)
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

# from PySide6.QtWidgets import QApplication
# import sys

# print("Before QApplication")
# app = QApplication(sys.argv)
# print("After QApplication")

# from sksurgeryvtk.widgets.vtk_overlay_window import VTKOverlayWindow

# print("Imported VTKOverlayWindow")

# window = VTKOverlayWindow()
# print("Created VTKOverlayWindow instance")
# window.show()

# sys.exit(app.exec())