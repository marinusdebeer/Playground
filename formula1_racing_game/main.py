"""
Enhanced 3D Racing Game using Panda3D

Instructions:

1. Install Panda3D:
   - Run `pip install panda3d`

2. Save this script as `enhanced_racing_game.py`

3. Run the game:
   - Run `python enhanced_racing_game.py`

Controls:
- W: Accelerate forward
- S: Accelerate backward
- A: Turn left
- D: Turn right
- Mouse Movement: Adjust camera view
- Scroll Wheel: Zoom in/out
- ESC: Exit the game
"""

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight, DirectionalLight, VBase4, Vec3, Point3,
    CollisionTraverser, CollisionNode, CollisionHandlerPusher,
    CollisionBox, BitMask32, NodePath, CardMaker, GeomVertexData,
    GeomVertexFormat, GeomVertexWriter, GeomTriangles, Geom, GeomNode,
    TextNode
)
from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
import sys
import math

class EnhancedRacingGame(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Disable default mouse camera controls
        self.disableMouse()

        # Set up the camera
        self.camera.reparentTo(render)

        # Create the racetrack
        self.create_track()

        # Create the car
        self.create_car()

        # Setup key controls
        self.accept("escape", sys.exit)
        self.accept("w", self.set_key, ["forward", True])
        self.accept("w-up", self.set_key, ["forward", False])
        self.accept("s", self.set_key, ["backward", True])
        self.accept("s-up", self.set_key, ["backward", False])
        self.accept("a", self.set_key, ["left", True])
        self.accept("a-up", self.set_key, ["left", False])
        self.accept("d", self.set_key, ["right", True])
        self.accept("d-up", self.set_key, ["right", False])

        # Mouse controls
        self.accept("wheel_up", self.zoom_in)
        self.accept("wheel_down", self.zoom_out)

        self.keys = {"forward": False, "backward": False, "left": False, "right": False}

        # Car movement variables
        self.speed = 0
        self.max_speed = 50
        self.acceleration = 30  # Units per second squared
        self.turn_rate = 100  # Degrees per second
        self.deceleration = 20  # Units per second squared

        # Camera control variables
        self.cam_distance = 10
        self.cam_height = 5
        self.mouse_sensitivity = 0.2
        self.cam_heading = 0
        self.cam_pitch = -10  # Slightly looking down
        self.is_mouse_down = False

        # Setup collision detection
        self.cTrav = CollisionTraverser()
        self.pusher = CollisionHandlerPusher()
        self.carCollider = self.car.attachNewNode(CollisionNode('car'))
        self.carCollider.node().addSolid(CollisionBox(Point3(0, 0, 0.5), 1, 2, 0.5))
        self.carCollider.node().setFromCollideMask(BitMask32.bit(1))
        self.carCollider.node().setIntoCollideMask(BitMask32.allOff())
        self.pusher.addCollider(self.carCollider, self.car)
        self.cTrav.addCollider(self.carCollider, self.pusher)

        # Task to update the game every frame
        self.taskMgr.add(self.update, "update")

        # Add some lighting
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor(VBase4(0.5, 0.5, 0.5, 1))
        render.setLight(render.attachNewNode(ambientLight))

        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(Vec3(-5, -5, -5))
        directionalLight.setColor(VBase4(0.7, 0.7, 0.7, 1))
        render.setLight(render.attachNewNode(directionalLight))

        # Set up the speedometer
        self.speedometer = OnscreenText(
            text="Speed: 0",
            pos=(-1.3, 0.9),
            scale=0.07,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )

        # Mouse controls
        self.accept("mouse1", self.set_mouse_button, [True])
        self.accept("mouse1-up", self.set_mouse_button, [False])

        # Center the mouse cursor
        self.win.movePointer(0, int(self.win.getProperties().getXSize() / 2),
                             int(self.win.getProperties().getYSize() / 2))

    def set_key(self, key, value):
        self.keys[key] = value

    def set_mouse_button(self, down):
        self.is_mouse_down = down

    def zoom_in(self):
        self.cam_distance -= 1
        if self.cam_distance < 5:
            self.cam_distance = 5

    def zoom_out(self):
        self.cam_distance += 1
        if self.cam_distance > 30:
            self.cam_distance = 30

    def create_cube(self, size=1, color=(1,1,1,1)):
        format = GeomVertexFormat.getV3n3()
        vdata = GeomVertexData('cube', format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')

        # Define cube vertices and normals
        s = size / 2.0
        vertices = [
            (-s, -s, -s),
            ( s, -s, -s),
            ( s,  s, -s),
            (-s,  s, -s),
            (-s, -s,  s),
            ( s, -s,  s),
            ( s,  s,  s),
            (-s,  s,  s),
        ]

        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (1, 2, 6, 5),
            (0, 3, 7, 4),
        ]

        normals = [
            (0, 0, -1),
            (0, 0, 1),
            (0, -1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (-1, 0, 0),
        ]

        triangles = GeomTriangles(Geom.UHStatic)
        for face_index, face in enumerate(faces):
            normal_vec = normals[face_index]
            idxs = []
            for vertex_index in face:
                idx = vertex.getWriteRow()
                vertex.addData3(*vertices[vertex_index])
                normal.addData3(*normal_vec)
                idxs.append(idx)
            triangles.addVertices(idxs[0], idxs[1], idxs[2])
            triangles.addVertices(idxs[0], idxs[2], idxs[3])

        geom = Geom(vdata)
        geom.addPrimitive(triangles)
        node = GeomNode('cube')
        node.addGeom(geom)
        cube = NodePath(node)
        cube.setColor(*color)
        return cube

    def create_track(self):
        # Create ground
        cm = CardMaker("ground")
        cm.setFrame(-100, 100, -100, 100)
        ground = render.attachNewNode(cm.generate())
        ground.setPos(0, 0, 0)
        ground.setHpr(0, -90, 0)
        ground.setColor(0, 1, 0, 1)  # Green ground
        ground.setCollideMask(BitMask32.bit(1))

        # Create boundary walls
        wall_thickness = 1
        wall_height = 2
        wall_length = 200
        walls = NodePath("walls")

        # Create walls around the track
        for x in [-100 + wall_thickness / 2, 100 - wall_thickness / 2]:
            wall = self.create_wall(wall_thickness, wall_length, wall_height)
            wall.reparentTo(walls)
            wall.setPos(x, 0, wall_height / 2)
        for y in [-100 + wall_thickness / 2, 100 - wall_thickness / 2]:
            wall = self.create_wall(wall_length, wall_thickness, wall_height)
            wall.reparentTo(walls)
            wall.setPos(0, y, wall_height / 2)
        walls.reparentTo(render)

        # Add checkpoints (simple lines on the ground)
        self.checkpoints = []
        for i in range(1, 5):
            cm = CardMaker(f"checkpoint_{i}")
            cm.setFrame(-10, 10, -1, 1)
            checkpoint = render.attachNewNode(cm.generate())
            checkpoint.setPos(0, i * 40 - 80, 0.01)
            checkpoint.setHpr(0, -90, 0)
            checkpoint.setColor(1, 1, 0, 1)  # Yellow lines
            checkpoint.setCollideMask(BitMask32.bit(1))
            self.checkpoints.append(checkpoint)

        # Initialize lap variables
        self.current_checkpoint = 0
        self.lap = 1
        self.total_laps = 3
        self.lap_text = OnscreenText(
            text=f"Lap: {self.lap}/{self.total_laps}",
            pos=(-1.3, 0.8),
            scale=0.07,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )

    def create_wall(self, x_size, y_size, z_size):
        wall = self.create_cube(size=1)
        wall.setScale(x_size, y_size, z_size)
        wall.setColor(1, 0, 0, 1)  # Red walls
        wall.setCollideMask(BitMask32.bit(1))
        return wall

    def create_car(self):
        self.car = NodePath("car")
        self.car.reparentTo(render)

        # Car body
        body = self.create_cube(size=2, color=(0, 0, 1, 1))  # Blue body
        body.reparentTo(self.car)
        body.setZ(0.5)

        # Wheels
        wheel_positions = [
            (0.9, 1.5, 0),
            (-0.9, 1.5, 0),
            (0.9, -1.5, 0),
            (-0.9, -1.5, 0)
        ]
        for pos in wheel_positions:
            wheel = self.create_cube(size=0.5, color=(0, 0, 0, 1))  # Black wheels
            wheel.reparentTo(self.car)
            wheel.setPos(*pos)

        # Initialize car position and orientation
        self.car.setPos(0, -80, 1)
        self.car.setH(0)

    def update(self, task):
        dt = globalClock.getDt()

        # Update speed based on acceleration/deceleration
        if self.keys["forward"]:
            self.speed += self.acceleration * dt
        elif self.keys["backward"]:
            self.speed -= self.acceleration * dt
        else:
            # Decelerate to zero speed
            if self.speed > 0:
                self.speed -= self.deceleration * dt
                if self.speed < 0:
                    self.speed = 0
            elif self.speed < 0:
                self.speed += self.deceleration * dt
                if self.speed > 0:
                    self.speed = 0

        # Clamp speed
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        elif self.speed < -self.max_speed / 2:
            self.speed = -self.max_speed / 2  # Reverse speed is half of forward speed

        # Update position based on speed
        distance = self.speed * dt
        self.car.setY(self.car, distance)

        # Update rotation based on turning and current speed
        if self.keys["left"]:
            turn_amount = self.turn_rate * dt * (self.speed / self.max_speed)
            self.car.setH(self.car.getH() + turn_amount)
        if self.keys["right"]:
            turn_amount = -self.turn_rate * dt * (self.speed / self.max_speed)
            self.car.setH(self.car.getH() + turn_amount)

        # Mouse camera control
        if self.is_mouse_down and self.mouseWatcherNode.hasMouse():
            # Get the mouse movement
            dx = self.mouseWatcherNode.getMouseX() * self.mouse_sensitivity
            dy = self.mouseWatcherNode.getMouseY() * self.mouse_sensitivity

            self.cam_heading -= dx * 100 * dt
            self.cam_pitch += dy * 100 * dt

            # Clamp the pitch
            if self.cam_pitch > 45:
                self.cam_pitch = 45
            elif self.cam_pitch < -10:
                self.cam_pitch = -10

            # Reset the mouse cursor to the center
            self.win.movePointer(0, int(self.win.getProperties().getXSize() / 2),
                                 int(self.win.getProperties().getYSize() / 2))

        # Update camera to follow the car smoothly
        angle_rad = math.radians(self.cam_heading + self.car.getH())
        cam_x = self.car.getX() - self.cam_distance * math.sin(angle_rad)
        cam_y = self.car.getY() - self.cam_distance * math.cos(angle_rad)
        cam_z = self.car.getZ() + self.cam_height + self.cam_distance * math.tan(math.radians(self.cam_pitch))

        self.camera.setPos(cam_x, cam_y, cam_z)
        self.camera.lookAt(self.car.getX(), self.car.getY(), self.car.getZ() + 1)

        # Update speedometer
        self.speedometer.setText(f"Speed: {int(abs(self.speed))}")

        # Check for lap completion
        self.check_lap()

        return Task.cont

    def check_lap(self):
        # Simple checkpoint system
        checkpoint = self.checkpoints[self.current_checkpoint]
        car_pos = self.car.getPos()

        # If the car crosses the checkpoint
        checkpoint_pos = checkpoint.getPos()
        distance = (Vec3(car_pos) - Vec3(checkpoint_pos)).length()

        if distance < 5:
            self.current_checkpoint += 1
            if self.current_checkpoint >= len(self.checkpoints):
                self.current_checkpoint = 0
                self.lap += 1
                if self.lap > self.total_laps:
                    self.lap = self.total_laps
                    self.game_over()
                else:
                    self.lap_text.setText(f"Lap: {self.lap}/{self.total_laps}")

    def game_over(self):
        # Display Game Over message
        self.game_over_text = OnscreenText(
            text="Race Completed!",
            pos=(0, 0),
            scale=0.2,
            fg=(1, 1, 0, 1),
            align=TextNode.ACenter
        )
        # Stop the car
        self.speed = 0
        # Remove update task
        self.taskMgr.remove("update")

if __name__ == "__main__":
    game = EnhancedRacingGame()
    game.run()
