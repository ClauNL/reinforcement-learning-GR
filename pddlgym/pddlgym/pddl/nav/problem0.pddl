(define (problem ipc-grid-10-10-10)

(:domain grid)
(:objects
place_0_0 - place
place_0_1 - place
place_0_2 - place
place_0_3 - place
place_0_4 - place
place_0_5 - place
place_0_6 - place
place_0_7 - place
place_0_8 - place
place_0_9 - place
place_1_0 - place
place_1_1 - place
place_1_2 - place
place_1_3 - place
place_1_4 - place
place_1_5 - place
place_1_6 - place
place_1_7 - place
place_1_8 - place
place_1_9 - place
place_2_0 - place
place_2_1 - place
place_2_2 - place
place_2_3 - place
place_2_4 - place
place_2_5 - place
place_2_6 - place
place_2_7 - place
place_2_8 - place
place_2_9 - place
place_3_0 - place
place_3_1 - place
place_3_2 - place
place_3_3 - place
place_3_4 - place
place_3_5 - place
place_3_6 - place
place_3_7 - place
place_3_8 - place
place_3_9 - place
place_4_0 - place
place_4_1 - place
place_4_2 - place
place_4_3 - place
place_4_4 - place
place_4_5 - place
place_4_6 - place
place_4_7 - place
place_4_8 - place
place_4_9 - place
place_5_0 - place
place_5_1 - place
place_5_2 - place
place_5_3 - place
place_5_4 - place
place_5_5 - place
place_5_6 - place
place_5_7 - place
place_5_8 - place
place_5_9 - place
place_6_0 - place
place_6_1 - place
place_6_2 - place
place_6_3 - place
place_6_4 - place
place_6_5 - place
place_6_6 - place
place_6_7 - place
place_6_8 - place
place_6_9 - place
place_7_0 - place
place_7_1 - place
place_7_2 - place
place_7_3 - place
place_7_4 - place
place_7_5 - place
place_7_6 - place
place_7_7 - place
place_7_8 - place
place_7_9 - place
place_8_0 - place
place_8_1 - place
place_8_2 - place
place_8_3 - place
place_8_4 - place
place_8_5 - place
place_8_6 - place
place_8_7 - place
place_8_8 - place
place_8_9 - place
place_9_0 - place
place_9_1 - place
place_9_2 - place
place_9_3 - place
place_9_4 - place
place_9_5 - place
place_9_6 - place
place_9_7 - place
place_9_8 - place
place_9_9 - place


)
(:init
(conn place_0_0 place_0_1) (conn place_0_0 place_1_0)
(conn place_0_1 place_0_0) (conn place_0_1 place_0_2) (conn place_0_1 place_1_1)
(conn place_0_2 place_0_1) (conn place_0_2 place_0_3) (conn place_0_2 place_1_2)
(conn place_0_3 place_0_2) (conn place_0_3 place_0_4) (conn place_0_3 place_1_3)
(conn place_0_4 place_0_3) (conn place_0_4 place_0_5)
(conn place_0_5 place_0_4) (conn place_0_5 place_0_6) (conn place_0_5 place_1_5)
(conn place_0_6 place_0_5) (conn place_0_6 place_0_7) (conn place_0_6 place_1_6)
(conn place_0_7 place_0_6) (conn place_0_7 place_0_8)
(conn place_0_8 place_0_7) (conn place_0_8 place_0_9) (conn place_0_8 place_1_8)
(conn place_0_9 place_0_8)
(conn place_1_0 place_1_1) (conn place_1_0 place_0_0) (conn place_1_0 place_2_0)
(conn place_1_1 place_1_0) (conn place_1_1 place_1_2) (conn place_1_1 place_0_1)
(conn place_1_2 place_1_1) (conn place_1_2 place_1_3) (conn place_1_2 place_0_2)
(conn place_1_3 place_1_2) (conn place_1_3 place_1_4) (conn place_1_3 place_0_3)
(conn place_1_4 place_1_3) (conn place_1_4 place_1_5)
(conn place_1_5 place_1_4) (conn place_1_5 place_1_6) (conn place_1_5 place_0_5)
(conn place_1_6 place_1_5) (conn place_1_6 place_1_7) (conn place_1_6 place_0_6)
(conn place_1_7 place_1_6) (conn place_1_7 place_1_8)
(conn place_1_8 place_1_7) (conn place_1_8 place_1_9) (conn place_1_8 place_0_8)
(conn place_1_9 place_1_8)
(conn place_2_0 place_2_1) (conn place_2_0 place_1_0) (conn place_2_0 place_3_0)
(conn place_2_1 place_2_0) (conn place_2_1 place_2_2) (conn place_2_1 place_3_1)
(conn place_2_2 place_2_1) (conn place_2_2 place_2_3)
(conn place_2_3 place_2_2) (conn place_2_3 place_2_4) (conn place_2_3 place_3_3)
(conn place_2_4 place_2_3) (conn place_2_4 place_2_5)
(conn place_2_5 place_2_4) (conn place_2_5 place_2_6)
(conn place_2_6 place_2_5) (conn place_2_6 place_2_7) (conn place_2_6 place_3_6)
(conn place_2_7 place_2_6) (conn place_2_7 place_2_8)
(conn place_2_8 place_2_7) (conn place_2_8 place_2_9) (conn place_2_8 place_3_8)
(conn place_2_9 place_2_8)
(conn place_3_0 place_3_1) (conn place_3_0 place_2_0) (conn place_3_0 place_4_0)
(conn place_3_1 place_3_0) (conn place_3_1 place_3_2) (conn place_3_1 place_2_1)
(conn place_3_2 place_3_1) (conn place_3_2 place_3_3)
(conn place_3_3 place_3_2) (conn place_3_3 place_3_4) (conn place_3_3 place_2_3)
(conn place_3_4 place_3_3) (conn place_3_4 place_3_5)
(conn place_3_5 place_3_4) (conn place_3_5 place_3_6)
(conn place_3_6 place_3_5) (conn place_3_6 place_3_7) (conn place_3_6 place_2_6)
(conn place_3_7 place_3_6) (conn place_3_7 place_3_8)
(conn place_3_8 place_3_7) (conn place_3_8 place_3_9) (conn place_3_8 place_2_8)
(conn place_3_9 place_3_8)
(conn place_4_0 place_4_1) (conn place_4_0 place_3_0) (conn place_4_0 place_5_0)
(conn place_4_1 place_4_0) (conn place_4_1 place_4_2)
(conn place_4_2 place_4_1) (conn place_4_2 place_4_3) (conn place_4_2 place_5_2)
(conn place_4_3 place_4_2) (conn place_4_3 place_4_4)
(conn place_4_4 place_4_3) (conn place_4_4 place_4_5) (conn place_4_4 place_5_4)
(conn place_4_5 place_4_4) (conn place_4_5 place_4_6)
(conn place_4_6 place_4_5) (conn place_4_6 place_4_7) (conn place_4_6 place_5_6)
(conn place_4_7 place_4_6) (conn place_4_7 place_4_8)
(conn place_4_8 place_4_7) (conn place_4_8 place_4_9)
(conn place_4_9 place_4_8)
(conn place_5_0 place_5_1) (conn place_5_0 place_4_0) (conn place_5_0 place_6_0)
(conn place_5_1 place_5_0) (conn place_5_1 place_5_2)
(conn place_5_2 place_5_1) (conn place_5_2 place_5_3) (conn place_5_2 place_4_2)
(conn place_5_3 place_5_2) (conn place_5_3 place_5_4)
(conn place_5_4 place_5_3) (conn place_5_4 place_5_5) (conn place_5_4 place_4_4)
(conn place_5_5 place_5_4) (conn place_5_5 place_5_6)
(conn place_5_6 place_5_5) (conn place_5_6 place_5_7) (conn place_5_6 place_4_6)
(conn place_5_7 place_5_6) (conn place_5_7 place_5_8)
(conn place_5_8 place_5_7) (conn place_5_8 place_5_9)
(conn place_5_9 place_5_8)
(conn place_6_0 place_6_1) (conn place_6_0 place_5_0) (conn place_6_0 place_7_0)
(conn place_6_1 place_6_0) (conn place_6_1 place_6_2)
(conn place_6_2 place_6_1) (conn place_6_2 place_6_3) (conn place_6_2 place_7_2)
(conn place_6_3 place_6_2) (conn place_6_3 place_6_4) (conn place_6_3 place_7_3)
(conn place_6_4 place_6_3) (conn place_6_4 place_6_5)
(conn place_6_5 place_6_4) (conn place_6_5 place_6_6) (conn place_6_5 place_7_5)
(conn place_6_6 place_6_5) (conn place_6_6 place_6_7) (conn place_6_6 place_7_6)
(conn place_6_7 place_6_6) (conn place_6_7 place_6_8) (conn place_6_7 place_7_7)
(conn place_6_8 place_6_7) (conn place_6_8 place_6_9)
(conn place_6_9 place_6_8)
(conn place_7_0 place_7_1) (conn place_7_0 place_6_0) (conn place_7_0 place_8_0)
(conn place_7_1 place_7_0) (conn place_7_1 place_7_2)
(conn place_7_2 place_7_1) (conn place_7_2 place_7_3) (conn place_7_2 place_6_2)
(conn place_7_3 place_7_2) (conn place_7_3 place_7_4) (conn place_7_3 place_6_3)
(conn place_7_4 place_7_3) (conn place_7_4 place_7_5)
(conn place_7_5 place_7_4) (conn place_7_5 place_7_6) (conn place_7_5 place_6_5)
(conn place_7_6 place_7_5) (conn place_7_6 place_7_7) (conn place_7_6 place_6_6)
(conn place_7_7 place_7_6) (conn place_7_7 place_7_8) (conn place_7_7 place_6_7)
(conn place_7_8 place_7_7) (conn place_7_8 place_7_9)
(conn place_7_9 place_7_8)
(conn place_8_0 place_8_1) (conn place_8_0 place_7_0) (conn place_8_0 place_9_0)
(conn place_8_1 place_8_0) (conn place_8_1 place_8_2)
(conn place_8_2 place_8_1) (conn place_8_2 place_8_3) (conn place_8_2 place_9_2)
(conn place_8_3 place_8_2) (conn place_8_3 place_8_4)
(conn place_8_4 place_8_3) (conn place_8_4 place_8_5) (conn place_8_4 place_9_4)
(conn place_8_5 place_8_4) (conn place_8_5 place_8_6)
(conn place_8_6 place_8_5) (conn place_8_6 place_8_7)
(conn place_8_7 place_8_6) (conn place_8_7 place_8_8) (conn place_8_7 place_9_7)
(conn place_8_8 place_8_7) (conn place_8_8 place_8_9) (conn place_8_8 place_9_8)
(conn place_8_9 place_8_8)
(conn place_9_0 place_9_1) (conn place_9_0 place_8_0)
(conn place_9_1 place_9_0) (conn place_9_1 place_9_2)
(conn place_9_2 place_9_1) (conn place_9_2 place_9_3) (conn place_9_2 place_8_2)
(conn place_9_3 place_9_2) (conn place_9_3 place_9_4)
(conn place_9_4 place_9_3) (conn place_9_4 place_9_5) (conn place_9_4 place_8_4)
(conn place_9_5 place_9_4) (conn place_9_5 place_9_6)
(conn place_9_6 place_9_5) (conn place_9_6 place_9_7)
(conn place_9_7 place_9_6) (conn place_9_7 place_9_8) (conn place_9_7 place_8_7)
(conn place_9_8 place_9_7) (conn place_9_8 place_9_9) (conn place_9_8 place_8_8)
(conn place_9_9 place_9_8)


(wall place_0_6)
(is-goal place_0_2)

(at-robot place_9_9)


)
(:goal
(and 
      (at-robot place_0_2)
)
))