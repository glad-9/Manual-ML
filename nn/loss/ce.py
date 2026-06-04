class CE:
    name = "ce"

    def forward(self, y_hat, y):
        return y_hat.forward(y)