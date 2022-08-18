(define (problem blocks)
    (:domain blocks)
    (:objects 
        d - block
        b - block
        a - block
        c - block
        e - block
        robot - robot
    )
    (:init 
        (clear c) 
        (clear a) 
        (clear d) 
        (ontable c) 
        (ontable b) 
        (ontable e)
        (on a b)
        (on d e)
        (handempty robot)

        ; action literals
        (pickup a)
        (putdown a)
        (unstack a)
        (stack a b)
        (stack a c)
        (stack a d)
        (stack a e)
        (pickup b)
        (putdown b)
        (unstack b)
        (stack b a)
        (stack b c)
        (stack b d)
        (stack b e)
        (pickup c)
        (putdown c)
        (unstack c)
        (stack c b)
        (stack c a)
        (stack c d)
        (stack c e)
        (pickup d)
        (putdown d)
        (unstack d)
        (stack d b)
        (stack d c)
        (stack d a)
        (stack d e)
        (pickup e)
        (putdown e)
        (unstack e)
        (stack e b)
        (stack e c)
        (stack e a)
        (stack e d)

    )
    (:goal (and (on b a) (on a c) (on d e) (clear d) (clear b)))
)
