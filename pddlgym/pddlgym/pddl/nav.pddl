(define (domain grid)
(:types place)
(:predicates (conn ?x - place ?y - place)
            (at-robot ?x - place)
            (wall ?x - place)
            (is-goal ?x - place)            
)



(:action move
:parameters (?curpos - place ?nextpos - place)
:precondition (and 
            (at-robot ?curpos) 
            (conn ?curpos ?nextpos) 
            (not (wall ?nextpos)))
:effect (and 
            (at-robot ?nextpos) 
            (not (at-robot ?curpos)))
)

