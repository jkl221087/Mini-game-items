import pygame
import sys
import random


class PygamePlay:
    def __init__(self):
        pygame.init()
        self.running = None
        self.H = 800
        self.W = 600
        self.WHITE = (255, 255, 255)
        self.BLACK = (34, 40, 49)
        self.RED = (255, 0, 0)
        self.ORANGE = (255, 165, 0)
        self.YELLOW = (255, 255, 0)
        self.GREEN = (0, 255, 0)

        self.background_color = self.BLACK
        self.screen = pygame.display.set_mode((800, 600))

        self.head_font = pygame.font.SysFont(None, 60)
        self.data = ['16x', '9x', '2x', '14x', '14x', '12x', '11x', '1x', '0.5x', '1x', '11x', '12x', '14x', '14x',
                     '2x', '9x', '16x']
        self.colors = [(255, 0, 0), (255, 102, 0), (255, 153, 0),  (255, 204, 0), (255, 255, 0), (255, 204, 0),
                       (255, 153, 0), (255, 102, 0), (255, 51, 0), (255, 0, 0), (255, 51, 0), (255, 102, 0),
                       (255, 153, 0), (255, 204, 0), (255, 255, 0)]

        self.dot_spacing = 30
        self.dot_radius = 3
        self.triangle_height = 15 * self.dot_spacing
        self.triangle_width = 15 * self.dot_spacing
        self.x_offset = 100

        self.block_width = 35
        self.block_height = 35
        self.block_spacing_x = 125
        self.block_spacing_y = 2
        self.font = pygame.font.Font(None, 20)
        self.coin_count = 1000

        self.input_box = pygame.Rect(100, 200, 140, 32)
        self.input_color = pygame.Color('lightskyblue3')
        self.active_color = pygame.Color('dodgerblue2')
        self.text_color = pygame.Color('WHITE')
        self.user_text = ''
        self.active = False

        self.bet_placed = False

        self.balls = []
        self.ball_radius = 5
        self.ball_speed = 1
        self.blocks = []
        self.points = []

    def point(self):
        self.points = []
        for row in range(4, 16):
            for col in range(row):
                x = (self.W // 2) - (row * self.dot_spacing // 2) + col * self.dot_spacing + self.x_offset
                y = (self.H // 2) - (self.triangle_height // 1.2) + row * self.dot_spacing
                pygame.draw.circle(self.screen, self.WHITE, (x, y), self.dot_radius)
                self.points.append((x, y))

    def block(self):
        self.blocks = []
        block_start_y = (self.H // 1.6)
        for i, (value, color) in enumerate(zip(self.data, self.colors)):
            rect_x = i * self.block_width + self.block_spacing_x
            rect_y = block_start_y + self.block_spacing_y
            pygame.draw.rect(self.screen, color, (rect_x, rect_y, self.block_width, self.block_height), border_radius=6)
            pygame.draw.rect(self.screen, self.BLACK, (rect_x, rect_y, self.block_width, self.block_height),
                             border_radius=6, width=2)
            text = self.font.render(value, True, self.BLACK)
            text_rect = text.get_rect(center=(rect_x + self.block_width // 2, rect_y + self.block_height // 2))
            self.screen.blit(text, text_rect)

            multiplier = float(value.strip('x'))  # 移除 'x' 並轉換為浮點數
            self.blocks.append((rect_x, rect_y, self.block_width, self.block_height, multiplier))

    def button(self):
        button_font = pygame.font.Font(None, 36)
        button_color = (255, 255, 255)
        button_active_color = (255, 0, 0)
        button_width = 80
        button_height = 50
        button_rect = pygame.Rect(100, 100, button_width, button_height)

        mouse_pos = pygame.mouse.get_pos()
        if button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, button_active_color, button_rect)
        else:
            pygame.draw.rect(self.screen, button_color, button_rect)

        # 繪製按鈕文字
        text = button_font.render("Bet", True, (0, 0, 0))
        text_rect = text.get_rect(center=button_rect.center)
        self.screen.blit(text, text_rect)

    def display_coins(self, coin_count):
        font = pygame.font.Font(None, 20)
        text_color = pygame.Color('white')
        text = font.render(f"Coins: {self.coin_count}", True, text_color)
        text_rect = text.get_rect()
        text_rect.topright = (self.W - 10, 10)
        self.screen.blit(text, text_rect)
        return coin_count

    def draw_input_box(self):
        # Draw the input box
        color = self.active_color if self.active else self.input_color
        pygame.draw.rect(self.screen, color, self.input_box, 2)
        txt_surface = self.font.render(self.user_text, True, self.text_color)
        self.screen.blit(txt_surface, (self.input_box.x + 5, self.input_box.y + 5))
        pygame.draw.rect(self.screen, color, self.input_box, 2)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                self.handle_key_press(event)

    def handle_mouse_click(self, pos):
        if self.input_box.collidepoint(pos):
            self.active = True
        else:
            self.active = False

        button_rect = pygame.Rect(100, 100, 80, 50)
        if button_rect.collidepoint(pos):
            self.place_bet()

    def handle_key_press(self, event):
        if self.active:
            if event.key == pygame.K_RETURN:
                try:
                    self.bet_amount = int(self.user_text)
                    if self.bet_amount > self.coin_count or self.bet_amount <= 0:
                        print("please enter a valid bet amount")
                        self.bet_amount = 0
                    else:
                        self.bet_placed = True
                    print(f"Bet amount set to: {self.bet_amount}")
                except ValueError:
                    print("Please enter a valid bet amount")
                    self.bet_amount = 0
                self.user_text = ''
            elif event.key == pygame.K_BACKSPACE:
                self.user_text = self.user_text[:-1]
            elif event.key == pygame.K_ESCAPE:
                self.user_text = ''
            else:
                self.user_text += event.unicode

    def place_bet(self):
        if 0 < self.bet_amount <= self.coin_count:
            self.coin_count -= self.bet_amount
            print(f"Bet placed: ${self.bet_amount}. Remaining coins: ${self.coin_count}")
            self.create_ball(self.bet_amount)
            self.bet_amount = 0
            self.bet_placed = False

    def create_ball(self, bet_amount):
        start_x = self.W // 2 + self.x_offset
        start_y = (self.H // 2) - (self.triangle_height // 1.2)
        self.balls.append([start_x, start_y, random.uniform(-0.5, 0.5),  self.ball_speed, bet_amount])
        print(self.balls)

    def update_balls(self):
        for ball in self.balls[:]:
            ball[0] += ball[2]
            ball[1] += ball[3]

            for point in self.points:
                if ((ball[0] - point[0]) ** 2 + (ball[1] - point[1]) ** 2) ** 0.5 < self.ball_radius + self.dot_radius:
                    ball[2] = random.uniform(-0.5, 0.5)

            for block in self.blocks:
                if (block[0] < ball[0] < block[0] + block[2] and
                        block[1] < ball[1] < block[1] + block[3]):
                    print(block[4])
                    bet_amount = ball[4]
                    print(bet_amount)
                    print(self.bet_amount)
                    reward = bet_amount * block[4]
                    print(reward)
                    print(f"Ball landed on {block[4]}x multiplier. Reward: {reward}")
                    self.coin_count += reward
                    self.balls.remove(ball)
                    break

            if ball[1] > self.H:
                self.balls.remove(ball)

            pygame.draw.circle(self.screen, self.WHITE, (int(ball[0]), int(ball[1])), self.ball_radius)

    def run(self):
        clock = pygame.time.Clock()
        while True:
            self.handle_events()
            self.screen.fill(self.background_color)
            self.point()
            self.block()
            self.display_coins(self.coin_count)
            self.button()
            self.draw_input_box()
            self.update_balls()
            pygame.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    game = PygamePlay()
    game.run()