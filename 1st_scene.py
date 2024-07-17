from manim import *
import manim

class GraphRepresentation(Scene):    

    def _build_matrix(self) -> VGroup:
        embs = [[Tex("e\\textsubscript{%d,%d}"%(i,j)) for i in range(3)] for j in range(3)]
        matrix = MobjectMatrix(embs , stretch_brackets=True)
        matrix_text = Tex("\\textnormal{Matrix With } $m\\times n$" , font_size=30).next_to(matrix, direction=DOWN)
        return VGroup(*[matrix , matrix_text])

    def _build_graph(self , vertices , edges) -> Graph:
        graph = Graph(vertices=vertices , edges=edges , labels=True, layout="circular" , layout_scale=2)
        return graph
    
    def _animate_graph(self, graph) -> None:
        pass

    def construct(self):
        title = Title("Data Representation Problem:")
        
        graph = self._build_graph([1 , 2 , 3 , 4 , 5] , [(1 , 2) , (4 , 3) , (2 , 3) , (2 , 5)])
        
        graph_text = Text("Normal Graph" , font_size=20 , weight=LIGHT)  
        graph_text.next_to(graph , direction=DOWN)

        graph_group = VGroup(*[graph_text , graph])
        matrix_group = self._build_matrix()
        
        self.play(Create(title , lag_ratio=0.05))
        self.play(Create(graph , lag_ratio=0.01))
        self.play(Create(graph_text , lag_ratio=0.001))
        self.wait()
        self.play(graph_group.animate(lag_ratio=0.02).shift(LEFT *  3))
        self.wait()
        matrix_group.shift(3 * RIGHT)
        self.play(Create(matrix_group , lag_ratio=0.001))
        self.wait()

        self.play(FadeOut(matrix_group , shift=RIGHT , lag_ratio=0.002) , FadeOut(graph_group , shift=LEFT , lag_ratio=0.002))
        self.play(Uncreate(title, lag_ratio=0.002))

class Word2vec(Scene):

    def construct(self):
        title = Title("Word2Vec Algorithm:")
        hello_word = Text("Hello!")

        list_embs = MobjectMatrix(
            [
                Tex("$\\alpha$\\textsubscript{1}") ,
                Tex("$\\alpha$\\textsubscript{2}") , 
                Tex("$\\alpha$\\textsubscript{3}") , 
                Tex("...") , 
                Tex("$\\alpha$\\textsubscript{n-1}") , 
                Tex("$\\alpha$\\textsubscript{n}") , 
            ]
        ).shift(RIGHT * 3)

        word2vec_text = Text("word2vec", font_size=20)
        conv_arrow = Arrow(start=LEFT , end=RIGHT , color=BLUE)
        word2vec_text.next_to(conv_arrow , direction=UP)

        conv_group = VGroup(*[conv_arrow , word2vec_text])

        sentience_text = Text("Hello World!" , font_size=70)
        sentience_text.generate_target()
        sentience_text.target.shift(3 * LEFT).scale(0.5)

        fs_fade_out = [FadeOut(conv_group ,shift = DOWN , lag_ratio=0.5) , FadeOut(list_embs , lag_ratio=0.5)]

        self.play(Create(title , lag_ratio=0.05))

        self.play(Create(hello_word , lag_ratio=0.002))
        self.play(hello_word.animate.shift(4 * LEFT))

        self.play(Create(conv_group , lag_ratio=0.05))

        self.play(Create(list_embs , lag_ratio=0.05))
        self.wait()

        self.play(*fs_fade_out)

        self.play(Transform(hello_word , sentience_text))
        self.wait()
        
        self.remove(hello_word)
        self.play(MoveToTarget(sentience_text))

        self.wait() 

        hello_vertex = "w1"
        world_vertex = "w2"

        linear_grahp = DiGraph([hello_vertex , world_vertex ] , [(hello_vertex , world_vertex)], layout="partite" , labels=True , label_fill_color=WHITE , vertex_config={
            "color" : BLUE,
        }, layout_config={
            "partitions" : [[hello_vertex] , [world_vertex]]
        }).next_to(sentience_text , direction=DOWN)
        self.play(Create(linear_grahp))

        self.wait()

        vertices = [1, 2, 3, 4]
        edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
        lt = {1: [0, 0, 0], 2: [1, 1, 0], 3: [1, -1, 0], 4: [-1, 0, 0]}
        G = Graph(vertices, edges, layout=lt , labels=True).move_to(RIGHT * 3)
        self.play(Create(G))

        self.wait()

        self.play(FadeOut(G , linear_grahp , sentience_text))
        self.play(Uncreate(title , lag_ratio=0.02))

        self.wait()



class Node2vec(Scene):
    def construct(self):
        title = Title("Node2Vec Algorithm:")
        graph = Graph(
            vertices=[1 , 2 , 3 , 4 , 5] ,
            edges=[(1 , 2) , (1 , 3) , (3 , 2)  , (2 , 5) , (3 , 5) , (4 , 5) , (2 ,4)] , 
            labels=True)
        graph_label = Text("Graph" , font_size=20).next_to(graph , DOWN)
        graph_group = VGroup(graph,  graph_label)
        
        
        spanned_graph = Graph(
            vertices=[1 , 2 , 3 , 4 , 5] ,
            edges=[(1 , 2) , (1 , 3) , (3 , 2)  , (2 , 5) , (3 , 5) , (4 , 5) , (2 ,4)] , 
            labels=True,
            layout="partite",
            partitions=[[1] , [3] , [2] , [5] , [4]])
        spanned_graph.move_to(3 * RIGHT)
        spanned_graph_label = Text("Spanning Tree" , font_size=20).next_to(spanned_graph , DOWN)
        span_graph_group = VGroup(spanned_graph_label , spanned_graph)

        complex_graph = Graph(
            vertices=[1 , 2 , 3 , 4 ,5 , 6],
            edges=((2 , 4) , (2 , 3) , (2 , 5) , (1,  3) , (1 , 2) , (1, 6) , (3 , 5) , (5 , 6) , (2 , 6)),
            vertex_config={
                "radius" : 0.2
            }
        )
        bfs_graph = complex_graph
        dfs_graph = complex_graph.copy()

        bfs_edge_config = {"color" : ORANGE}
        dfs_edge_config = {"color" : GREEN}

        homophily_label = Text("Homophily" , font_size=20).next_to(bfs_graph , direction=DOWN)
        structEqv_label = Text("Structural Equivalence" , font_size=20).next_to(dfs_graph, direction=DOWN)

        homophily_group = Group(homophily_label , bfs_graph)
        structEqv_group = Group(dfs_graph , structEqv_label)

        context_label = Text("Context??")
        context_measures = Text("Homophily & Structural Equivalence" , font_size=20 , t2w={"Homophily" : BOLD , "Structural Equivalence" : BOLD} , t2c={"Homophily" : ORANGE , "Structural Equivalence" : GREEN}).next_to(context_label , direction=DOWN)

        randomWalk_label= Text("Random Walk")
        bfs_dfs_label= Text("Breadth-First or Depth-First?" , t2c={"Breadth-First" : ORANGE , "Depth-First" : GREEN})

        randomWalkOperator_label= Text("random walk operator (q)" , color=YELLOW , font_size=20).next_to(randomWalk_label , direction=DOWN)

        transform_arrow = Arrow(max_stroke_width_to_length_ratio=0.5 )
        word2vec_label = Text("Word2Vec" , font_size=20).next_to(transform_arrow , direction=UP)
        word2vec_group = VGroup(transform_arrow , word2vec_label)

        embs_matrix = MobjectMatrix([
            [Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$1,1$}}}}") , Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$1,2$}}}}") , Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$...$}}}}")],
            [Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$2,1$}}}}") , Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$2,2$}}}}") , Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$...$}}}}")],
            [Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$3,1$}}}}") , Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$3,2$}}}}") , Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$...$}}}}")],
            [Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$4,1$}}}}") , Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$4,2$}}}}") , Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$...$}}}}")],
            [Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$5,1$}}}}") , Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$5,2$}}}}") , Tex("\\scriptsize{\\centerline{$emb$\\textsubscript{\\tiny{$...$}}}}")]
        ] , stretch_brackets=True , v_buff=1 , h_buff=1.5).move_to(4 * RIGHT)

        self.play(Create(title , lag_ratio=0.005))
        self.play(Create(graph , lag_ratio=0.003), Create(graph_label , lag_ratio=0.003))
        self.wait()

        self.play(graph.animate.add_edges((1 , 3) , edge_config={"color" : RED}), lag_ratio=0.001)
        self.play(graph.animate.add_edges((3 , 2) , edge_config={"color" : RED}), lag_ratio=0.001)
        self.play(graph.animate.add_edges((2 , 5) , edge_config={"color" : RED}), lag_ratio=0.001)
        self.play(graph.animate.add_edges((5 , 4) , edge_config={"color" : RED}), lag_ratio=0.001)
        self.wait()

        self.play(graph_group.animate.shift(4 * LEFT))
        self.wait()

        graph_copy = graph.copy()
        self.play(Transform(graph_copy, spanned_graph) , Create(spanned_graph_label) , lag_ratio=0.002)
        self.add(spanned_graph)
        self.remove(graph_copy)
        self.wait()


        self.play(FadeOut(graph_group , shift=RIGHT), FadeOut(span_graph_group , shift=LEFT))
        self.wait()

        self.play(Write(context_label))
        self.wait()

        self.play(Write(context_measures))
        self.wait()

        self.play(Unwrite(context_label),  Uncreate(context_measures))
        self.wait()

        self.play(Create(complex_graph))
        self.wait()

        self.play(dfs_graph.animate.shift(RIGHT * 3) , bfs_graph.animate.shift(LEFT * 3))
        self.wait()

        structEqv_label.shift(RIGHT * 3)
        homophily_label.shift(LEFT * 3)
        self.play(Create(structEqv_label , lag_ratio=0.005) ,Create(homophily_label , lag_ratio=0.005))
        self.wait()


        self.play(*[
            bfs_graph.animate.add_edges((2 , 1) , edge_config=bfs_edge_config),
            bfs_graph.animate.add_edges((2 , 3) , edge_config=bfs_edge_config),
            bfs_graph.animate.add_edges((2 , 5) , edge_config=bfs_edge_config),
            bfs_graph.animate.add_edges((2 , 6) , edge_config=bfs_edge_config),
            bfs_graph.animate.add_edges((2 , 4) , edge_config=bfs_edge_config)
        ] , lag_ratio=0.003)
        self.wait()

        self.play(*[
            dfs_graph.animate.add_edges((4 , 2) , edge_config=dfs_edge_config),
            dfs_graph.animate.add_edges((2 , 1) , edge_config=dfs_edge_config),
            dfs_graph.animate.add_edges((1 , 3) , edge_config=dfs_edge_config),
            dfs_graph.animate.add_edges((3 , 5) , edge_config=dfs_edge_config),
            dfs_graph.animate.add_edges((5 , 6) , edge_config=dfs_edge_config)
        ] , lag_ratio=0.003)
        self.wait()

        self.play(FadeOut(homophily_group , shift=LEFT) , FadeOut(structEqv_group , shift=RIGHT))
        self.wait()

        self.play(Write(bfs_dfs_label))
        self.wait()

        self.play(Transform(bfs_dfs_label , randomWalk_label))
        self.add(randomWalk_label)
        self.remove(bfs_dfs_label)
        self.wait()

        self.play(Write(randomWalkOperator_label))
        self.wait()

        self.play(FadeOut(randomWalkOperator_label) , FadeOut(randomWalk_label) , lag_ratio=0.05)

        span_graph_group.center().move_to(LEFT * 4)
        self.play(Create(span_graph_group) , lag_ratio=0.04)
        self.wait()

        self.play(Create(word2vec_group) , lag_ratio=0.04)
        self.wait()

        self.play(Create(embs_matrix) , lag_ratio=0.04)
        self.wait()

        self.play(FadeOut(span_graph_group) , FadeOut(word2vec_group) , FadeOut(embs_matrix) , lag_ratio=0.05)
        self.play(Unwrite(title))
        self.wait()