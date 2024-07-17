from manim import *

class Solution(Scene):
    def construct(self):
        title = Title("Solution:")
        predict_shortest_distance_text = Text("Model Can Predict Shortest Distance" , font_size=40 , t2c={"Distance" : GREEN_C})
        predict_shortest_path_text = Text(
            "But Can It Predict Shortest Path?" , 
            font_size= 30,
            t2c={"Path" : GOLD_A}).next_to(predict_shortest_distance_text , direction= DOWN)
        graph_type_question_text = Text("The Graph Type?" , color=RED_C).next_to(predict_shortest_distance_text , direction=UP)
        questions_group = VGroup(*[predict_shortest_distance_text , predict_shortest_path_text , graph_type_question_text])

        edges_table = MathTable([
            ["start" , "end" , "distance"],
            [1 , 2 , 5] , 
            [2 , 3 , 4] ,
            [2 , 4 , 3] , 
        ] , include_outer_lines =True).surround(Rectangle()).move_to(LEFT * 3)

        first_row_embs = MobjectMatrix([
            [
                MathTex("\\alpha\\textsubscript{1}") , 
                MathTex("\\alpha\\textsubscript{2}") , 
                MathTex("\\alpha\\textsubscript{3}") , 
                MathTex("\\alpha\\textsubscript{4}") ,
                MathTex("\\cdots") ,
                MathTex("\\alpha\\textsubscript{$m$}") ,
            ]
        ])

        second_row_embs = MobjectMatrix([
            [
                MathTex("\\beta\\textsubscript{1}") , 
                MathTex("\\beta\\textsubscript{2}") , 
                MathTex("\\beta\\textsubscript{3}") , 
                MathTex("\\beta\\textsubscript{4}") ,
                MathTex("\\cdots") ,
                MathTex("\\beta\\textsubscript{$m$}") ,
            ]
        ])

        third_row_embs = MobjectMatrix([
            [
                MathTex("\\gamma\\textsubscript{1}") , 
                MathTex("\\gamma\\textsubscript{2}") , 
                MathTex("\\gamma\\textsubscript{3}") , 
                MathTex("\\gamma\\textsubscript{4}") ,
                MathTex("\\cdots") ,
                MathTex("\\gamma\\textsubscript{$m$}") ,
            ]
        ])

        avg_rows_embs = MobjectMatrix([
            [
                MathTex("\\omega\\textsubscript{1}") , 
                MathTex("\\omega\\textsubscript{2}") , 
                MathTex("\\omega\\textsubscript{3}") , 
                MathTex("\\omega\\textsubscript{4}") ,
                MathTex("\\cdots") ,
                MathTex("\\omega\\textsubscript{$m$}") ,
            ]
        ])
        avg_embs = MathTex("\\omega\\textsubscript{$i$} = avg(\\alpha\\textsubscript{$i$} , \\beta\\textsubscript{$i$})").surround(Rectangle()).next_to(avg_rows_embs , direction=DOWN)


        rows_embs = VGroup(first_row_embs , second_row_embs , third_row_embs).arrange(buff=0.2, direction=DOWN).surround(Rectangle()).shift(3 * RIGHT)

        graph_type_answer_text = Text("Model does not care about graph type" , color=GREEN_B , font_size=30).next_to(avg_rows_embs,  direction=UP)


        knowing_path_question = Text("But How Can We Know The Shorest Path?" , font_size=30 , t2g={"Shortest Path" : color_gradient([YELLOW , GREEN] , 2)} , t2s={"Shortest Path" : ITALIC})
        a_star_text = Text("(A* Algorithm)" , slant=ITALIC , gradient=color_gradient((GOLD , GOLD_B) , 4)).next_to(knowing_path_question , direction=DOWN)

        a_star_graph = Graph(
            vertices=[1 , 2 , 3 , 4 , "dest"] , 
            edges=[(1 , 2) , (1 , 3) ,  (1 , 4) , (2 , "dest") , (3 ,  "dest") , (4 , "dest")] ,
            labels=True , 
            layout="partite",
            layout_scale=4,
            vertex_config= {
                "dest" : {"fill_color" : DARK_BLUE , "radius" : 1}, 
                "radius" : 1,
            },
            edge_config={
                (1 , 2) : {
                    "stroke_width" : 7
                },
                (1 , 3) : {
                    "stroke_width" : 7
                },
                (1 , 4) : {
                    "stroke_width" : 7
                },
            },
            partitions=[[1] , [2 , 3 ,4] , ["dest"]]
        ).surround(Rectangle()).shift(LEFT * 3)


        edge_h1 = DashedLine(a_star_graph.vertices[2] , a_star_graph.vertices["dest"] , stroke_width=3)
        edge_h2 = DashedLine(a_star_graph.vertices[3] , a_star_graph.vertices["dest"] , stroke_width=3)
        edge_h3 = DashedLine(a_star_graph.vertices[4] , a_star_graph.vertices["dest"] , stroke_width=3)

        a_star_graph.edges[(2 , "dest")] = edge_h1
        a_star_graph.edges[(3 , "dest")] = edge_h2
        a_star_graph.edges[(4 , "dest")] = edge_h3


        a_star_graph.edges[(2 , "dest")].set_color(DARK_BLUE)
        a_star_graph.edges[(3 , "dest")].set_color(DARK_BLUE)
        a_star_graph.edges[(4 , "dest")].set_color(DARK_BLUE)
        
        
        heuristic_table = MobjectTable([
            [Text("Vertex") , Text("Heuristic")],
            [Integer(1) , MathTex("h1")],
            [Integer(2) , MathTex("h2")],
            [Integer(3) , MathTex("h3")],
            [Integer(4) , MathTex("h4")],
        ]).surround(Rectangle()).shift(RIGHT * 3)

        final_dist_text = MathTex(
            """dist(1 , dest) = 
            \\min \\left\\{ 
            \\begin{array}{c}
            1 + h(2) \\\ 
            1 + h(3) \\\ 
            1 + h(4) \\end{array}
            \\right.
            """).surround(Rectangle(height=4 , width=6)).move_to(3 * DOWN).move_to(3 * DOWN)
        
        model_accuracy_text = Text("Model Accuracy is so Important" , font_size=30 , t2g={"Model Accuracy" : color_gradient((BLUE , GREEN) , 4)})
        
        

        self.play(Write(title), lag_ratio=0.002)
        self.play(Write(predict_shortest_distance_text) , lag_ratio=0.002)
        self.wait()
        self.play(Write(predict_shortest_path_text) , lag_ratio=0.002)
        self.wait()
        self.play(Write(graph_type_question_text), lag_ratio=0.002)
        self.wait()

        self.play(FadeOut(questions_group , shift=RIGHT) , lag_raito=0.002)
        self.wait()

        self.play(Write(edges_table) , lag_ratio=0.002)
        for i_row in range(len(edges_table.get_rows()[1:])):
            row_copy = edges_table.get_rows()[i_row + 1].copy()
            self.play(Transform(row_copy , rows_embs[i_row]) , lag_ratio=0.002)
            self.remove(row_copy)
            self.add(rows_embs[i_row])

        self.wait()

        self.play(FadeOut(edges_table , shift=LEFT) , rows_embs.animate.center() , lag_ratio=0.002)
        self.wait()

        self.play(Transform(rows_embs , avg_rows_embs) , lag_ratio=0.002)
        self.add(avg_rows_embs)
        self.remove(rows_embs)
        self.wait()
        self.play(Write(avg_embs))
        self.wait()
        
        self.play(Write(graph_type_answer_text) , lag_ratio=0.002)
        self.wait()
        self.play(avg_rows_embs.animate.set_color(GOLD))
        self.wait()

        self.play(FadeOut(avg_rows_embs , shift=LEFT) , FadeOut(avg_embs , shift=RIGHT) , FadeOut(graph_type_answer_text) , lag_ratio=0.002)
        self.wait()

        self.play(Write(knowing_path_question) , lag_ratio=0.002)
        self.wait()
        self.play(Write(a_star_text) , lag_ratio=0.002)
        self.wait()

        self.play(FadeOut(knowing_path_question) , lag_ratio=0.002)
        self.play(FadeOut(a_star_text) , lag_ratio=0.002)

        for vertex in a_star_graph.vertices:
            self.play(Write(a_star_graph[vertex]) , lag_ratio=0.002)

        self.play(Write(a_star_graph.edges[(1 , 2)]) , Write(a_star_graph.edges[(1 , 3)]) , Write(a_star_graph.edges[(1 , 4)]) , lag_ratio=0.002)
        self.wait()
        self.play(Write(a_star_graph.edges[(2 , "dest")]) , Write(a_star_graph.edges[(3 , "dest")]) , Write(a_star_graph.edges[(4 , "dest")]) , lag_ratio=0.002)
        self.wait()
        self.play(Write(heuristic_table))
        self.wait()

        self.play(Write(final_dist_text))
        self.wait()

        self.play(FadeOut(a_star_graph , shift=LEFT) , FadeOut(heuristic_table , shift=RIGHT) , FadeOut(final_dist_text , shift=DOWN) , FadeOut(*[edge_h3 , edge_h1 , edge_h2] , shift=LEFT))

        self.play(Write(model_accuracy_text))
        self.wait()
        
        self.play(Unwrite(model_accuracy_text))

        self.wait()
        
        self.play(Unwrite(title))



