import random



class HaS(object):
    """
    HaS class from Project 4

    Args:
        drop_rate(float): Desired drop rate in the number of blocks.
        grid_size(int): Desired size of grid
    """

    def __init__(self, grid_size, drop_rate, mean_value):
        self.mean_value = mean_value
        if not isinstance(grid_size, int):
            raise Exception("'grid_size' must be int")
        else:
            self.grid_size = grid_size
        if not isinstance(drop_rate, float):
            raise Exception("'drop_rate' must be float")
        else:
            self.drop_rate = drop_rate

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        if self.grid_size != 0:
            for x in range(0, w, self.grid_size):
                for y in range(0, h, self.grid_size):
                    x_end = min(w, x + self.grid_size)
                    y_end = min(h, y + self.grid_size)
                    if random.random() <= self.drop_rate:
                        img[0, x:x_end, y:y_end] = self.mean_value[0]
                        img[1, x:x_end, y:y_end] = self.mean_value[1]
                        img[2, x:x_end, y:y_end] = self.mean_value[2]

        return img

    def __repr__(self):
        return self.__class__.__name__ + 'grid_size={0}, drop_rate={1}'.format(self.grid_size, self.drop_rate)