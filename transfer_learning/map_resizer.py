from __future__ import annotations
import math
from typing import Tuple

class Coord:
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def distance(self, coord: Coord) -> float:
        return math.sqrt(math.pow((self.x - coord.x), 2) + math.pow((self.y - coord.y), 2))

class MapResizer:
    def __init__(self, map: list[str], size: int) -> None:
        self.map = map
        self.size = size
        self.start, self.goal = self._find_start_goal()
    
    def set_map(self, map:list[str]) -> None:
        self.map = map

    def set_size(self, size: int) -> None:
        self.size = size

    def _find_start_goal(self) -> Tuple[Coord, Coord]: 
        start = Coord()
        goal = Coord()
        found_start = False
        found_goal = False
        for y, row in enumerate(self.map):
            for x, pos in enumerate(row):
                if found_start and found_goal:
                    return [start, goal]
                if pos == "G":
                    goal.x = x
                    goal.y = y
                    found_goal = True
                    continue
                if pos == "S":
                    start.x = x
                    start.y = y
                    found_start = True
                    continue
        return [start, goal]    

    def _map_has_floor(self, map: list[str]) -> bool:
        for line in map:
            for pos in line:
                if pos == "F": return True
        return False
    
    def _map_has_start(self, map: list[str]) -> bool:
        for line in map:
            for pos in line:
                if pos == "S": return True
        return False

    def _map_has_goal(self, map: list[str]) -> bool:
        for line in map:
            for pos in line:
                if pos == "G": return True
        return False

    def _center_shruken_map(self) -> list[str]:
        shrunken_map = [''] * self.size
        for y in range(self.size):
            for x in range(self.size):
                if len(self.map)-1 >= (self.start.y + y-self.y_displacement):
                    if len(self.map[self.start.y + y-self.y_displacement]) - 1 >= (self.start.x+x-self.x_displacement):
                        shrunken_map[y] += self.map[(self.start.y + y) - self.y_displacement][(self.start.x + x) - self.x_displacement]
                        continue
                shrunken_map[y] += "H"
        return shrunken_map
    
    def _get_floor_coords(self, map: list[str]) -> list[Coord]:
        floor_coords = []
        for y, line in enumerate(map):
            for x, pos in enumerate(line):
                if pos == "F": 
                    floor_coords.append(Coord(x+(len(self.map)-self.size), y+(len(self.map)-self.size)))
        return floor_coords

    def _shrink_map(self) -> list[str]:
        shrunken_map = [''] * self.size

        self.x_displacement = 0
        self.y_displacement = 0
        # calculate displacement to keep shruken map within original tile bounds
        if (len(self.map[self.start.y]) - self.start.x) < self.size: self.x_displacement = abs(self.size - (len(self.map[self.start.y]) - self.start.x)) % self.size
        if (len(self.map) - self.start.y) < self.size: self.y_displacement = abs(self.size - (len(self.map) - self.start.y)) % self.size

        # shrink map that has at least one valid tile
        # this is because the goal will now replace the closest valid tile
        no_start = False
        shrunken_map = self._center_shruken_map()
        while not self._map_has_floor(shrunken_map):
            if self.x_displacement == self.size or self.y_displacement == self.size:
                break
            if self.x_displacement >= self.y_displacement: 
                self.x_displacement += 1
            else:
                self.y_displacement += 1
            shrunken_map = self._center_shruken_map()
            if not self._map_has_start(shrunken_map):
                if no_start == True:
                    if self.x_displacement >= self.y_displacement: 
                        self.x_displacement -= 1
                    else:
                        self.y_displacement -= 1
                    shrunken_map = self._center_shruken_map()
                    break
                else:
                    no_start = True
                    if self.x_displacement >= self.y_displacement: 
                        self.x_displacement -= 1
                    else:
                        self.y_displacement -= 1
                    shrunken_map = self._center_shruken_map()

        #print(f'{self.x_displacement} {self.y_displacement}')
        if not self._map_has_goal(shrunken_map):
            floor_coords = self._get_floor_coords(shrunken_map)
            min_distance = self.goal.distance(floor_coords[0])
            min_coord = floor_coords[0]
            for floor_coord in floor_coords:
                #print(floor_coord.x, floor_coord.y)
                if self.goal.distance(floor_coord) < min_distance:
                    min_coord = floor_coord
            min_coord_row = list(shrunken_map[min_coord.y-(len(self.map)-self.size)])
            min_coord_row[min_coord.x-(len(self.map)-self.size)] = 'G'
            shrunken_map[min_coord.y-(len(self.map)-self.size)] = ''.join(min_coord_row)
        
        return shrunken_map

    def _grow_map(self) -> list[str]:
        grown_map = [''] * self.size

        for i in range(self.size):
            if len(self.map)-1 >= i:
                while len(grown_map[i]) < self.size:
                    if len(self.map[i])-1 >= len(grown_map[i]):
                        grown_map[i] += self.map[i][len(grown_map[i])]
                    else:
                        grown_map[i] += 'H'
            else:
                grown_map[i] = 'H' * self.size

        return grown_map

    def convert_map(self) -> list[str]:
        if len(self.map) == self.size:
            return self.map
        elif len(self.map) > self.size:
            return self._shrink_map()
        else:
            return self._grow_map()